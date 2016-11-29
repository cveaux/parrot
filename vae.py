"""
Usage:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -u vae.py --n_frames 30 \
--weight_norm True --skip_conn True --dim 1024 --n_rnn 3 --bidirectional False \
--rnn_type GRU --learn_h0 False --batch_size 32 --kgmm 10 
"""



import datasets
import sys
import os

assert(os.environ['NN_LIB'])
sys.path.append(os.environ['NN_LIB'])

import lib
import lib.ops

from time import time
from datetime import datetime
print "Experiment started at:", datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M')
exp_start = time()

import os, sys, glob
sys.path.insert(1, os.getcwd())
import argparse

import numpy
numpy.random.seed(123)
np = numpy
import random
random.seed(123)

import theano
import theano.tensor as T
import theano.ifelse
import lasagne

from gmm_utils import cost_gmm, sample_gmm

def get_args():
    def t_or_f(arg):
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
           raise ValueError('Arg is neither `True` nor `False`')

    def check_non_negative(value):
        ivalue = int(value)
        if ivalue < 0:
             raise argparse.ArgumentTypeError("%s is not non-negative!" % value)
        return ivalue

    def check_positive(value):
        ivalue = int(value)
        if ivalue < 1:
             raise argparse.ArgumentTypeError("%s is not positive!" % value)
        return ivalue

    def check_unit_interval(value):
        fvalue = float(value)
        if fvalue < 0 or fvalue > 1:
             raise argparse.ArgumentTypeError("%s is not in [0, 1] interval!" % value)
        return fvalue

    # No default value here. Indicate every single arguement.
    parser = argparse.ArgumentParser(
        description='vae.py\nIndicate most of the arguments.')

    # Hyperparameter arguements:
    parser.add_argument('--n_frames', help='How many "frames" to include in each\
            Truncated BPTT pass', type=check_positive, required=True)
    parser.add_argument('--weight_norm', help='Adding learnable weight normalization\
            to all the linear layers (except for the embedding layer)',\
            type=t_or_f, required=True)
    parser.add_argument('--skip_conn', help='Add skip connections to RNN', type=t_or_f, required=True)
    parser.add_argument('--dim', help='Dimension of RNN',\
            type=check_positive, required=True)
    parser.add_argument('--n_rnn', help='Number of layers in the stacked RNN',
            type=check_positive, choices=xrange(1,6), required=True)
    parser.add_argument('--bidirectional', help='Whether to have bidirectional Encoder',
            type=t_or_f, required=True)
    parser.add_argument('--rnn_type', help='GRU or LSTM', choices=['LSTM', 'GRU'],\
            required=True)
    parser.add_argument('--learn_h0', help='Whether to learn the initial state of RNN',\
            type=t_or_f, required=True)
    parser.add_argument('--batch_size', help='size of mini-batch',
            type=check_positive, choices=[32, 64, 128, 256], required=True)

    parser.add_argument('--kgmm', help='Number of components in GMM',
            type=check_positive, required=True)

    parser.add_argument('--lr', help='Initial learning rate',
            type=lib.floatX, default = lib.floatX(0.001))

    parser.add_argument('--grad_clip', help='Upper limit on gradient',
            type=lib.floatX, default = lib.floatX(1.))

    parser.add_argument('--resume', help='Resume the same model from the last checkpoint.',\
            required=False, default=False, action='store_true')

    args = parser.parse_args()

    # NEW
    # Create tag for this experiment based on passed args
    tag_str_list = []
    for arg_name in sorted(args.__dict__.keys()):
        if arg_name != "resume":
            tag_str_list.append("{}_{}".format(arg_name, args.__dict__[arg_name]))

    tag = "_".join(tag_str_list)

    print "Created experiment tag for these args:"
    print tag

    return args, tag

args, tag = get_args()

FOLDER_PREFIX = '/Tmp/kumarkun/vocoder_vae'
GRAD_CLIP = args.grad_clip
LEARNING_RATE = args.lr

N_FRAMES = args.n_frames # How many 'frames' to include in each truncated BPTT pass
WEIGHT_NORM = args.weight_norm
SKIP_CONN = args.skip_conn
DIM = args.dim # Model dimensionality.
N_RNN = args.n_rnn # How many RNNs to stack
BIDIRECTIONAL = args.bidirectional
RNN_TYPE = args.rnn_type
H0_MULT = 2 if RNN_TYPE == 'LSTM' else 1
LEARN_H0 = args.learn_h0

BATCH_SIZE = args.batch_size
RESUME = args.resume
EPS = 1e-5
VOCODER_DIM = 63
INPUT_DIM = 400
OUTPUT_DIM = VOCODER_DIM

K_GMM = args.kgmm

print os.environ['FUEL_DATA_PATH']

def Encoder(speech, h0):
    """
    Create inference model to infer one single latent variable using bidirectional GRU followed by non-causal dilated convolutions
    """
    if RNN_TYPE == 'GRU':
        rnns_out, last_hidden = lib.ops.stackedGRU('Encoder.GRU',
                                                   N_RNN,
                                                   VOCODER_DIM,
                                                   DIM,
                                                   speech,
                                                   h0=h0,
                                                   weightnorm=WEIGHT_NORM,
                                                   skip_conn=SKIP_CONN)
    elif RNN_TYPE == 'LSTM':
        rnns_out, last_hidden = lib.ops.stackedLSTM('Encoder.LSTM',
                                                    N_RNN,
                                                    VOCODER_DIM,
                                                    DIM,
                                                    speech,
                                                    h0=h0,
                                                    weightnorm=WEIGHT_NORM,
                                                    skip_conn=SKIP_CONN)

    return rnns_out.mean(axis = 1)

def decoder(latent_var, text_features, h0, reset):
    """
    TODO: Change it to use latent varibales
    For now, only use text text features
    """
    learned_h0 = lib.param(
        'Decoder.h0',
        numpy.zeros((N_RNN, H0_MULT*DIM), dtype=theano.config.floatX)
    )
    # Handling LEARN_H0
    learned_h0.param = LEARN_H0
    learned_h0 = T.alloc(learned_h0, h0.shape[0], N_RNN, H0_MULT*DIM)
    learned_h0 = T.unbroadcast(learned_h0, 0, 1, 2)
    h0 = theano.ifelse.ifelse(reset, learned_h0, h0)

    rnns_out, last_hidden = lib.ops.stackedLSTM('Decoder.LSTM',
                                                    N_RNN,
                                                    INPUT_DIM,
                                                    DIM,
                                                    text_features,
                                                    h0=h0,
                                                    weightnorm=WEIGHT_NORM,
                                                    skip_conn=SKIP_CONN)

    output = lib.ops.Linear(
        'Decoder.Output',
        DIM,
        (2* OUTPUT_DIM + 1)*K_GMM,
        rnns_out,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )

    mu = output[:,:OUTPUT_DIM*K_GMM]
    sig = T.exp(output[:,OUTPUT_DIM*K_GMM : 2*OUTPUT_DIM*K_GMM]) + lib.floatX(EPS)
    weights = output[:,-K_GMM:]

    return mu, sig, weights, last_hidden




text_features = T.tensor3('text_features')
vocoder_audio = T.tensor3('vocoder_audio')
h0        = T.tensor3('h0')
reset     = T.iscalar('reset')
mask      = T.matrix('mask')


mu, sigma, weights, last_hidden = decoder(None, text_features, h0, reset)

cost = cost_gmm(vocoder_audio, mu, sigma, weights)

cost = T.sum(cost * mask)/mask.sum()

params = lib.get_params(cost, lambda x: hasattr(x, 'param') and x.param==True)
lib.print_params_info(params, path=FOLDER_PREFIX)

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

updates = lasagne.updates.adam(grads, params, learning_rate=LEARNING_RATE)


train_fn = theano.function(
    [text_features, h0, reset, mask, vocoder_audio],
    [cost, last_hidden],
    updates = updates
)

data_stream = datasets.parrot_stream('vctk', True)

import ipdb; ipdb.set_trace()
