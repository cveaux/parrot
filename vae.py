"""
Usage:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -u vae.py --n_frames 50 \
--weight_norm True --skip_conn False --dim 512 --n_rnn 3 --bidirectional False \
--rnn_type GRU --learn_h0 False --batch_size 8 --kgmm 20 --dataset vctk --ldim 64
"""

import sys
import os

assert(os.environ['NN_LIB'])
sys.path.insert(1, os.environ['NN_LIB'])

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
import itertools

from theano.sandbox.rng_mrg import MRG_RandomStreams

import datasets
from gmm_utils import cost_gmm, sample_gmm
from generate import generate_wav

theano_rng = MRG_RandomStreams(457)

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
            type=check_positive, choices=[8, 16, 32, 64, 128, 256], required=True)

    parser.add_argument('--kgmm', help='Number of components in GMM',
            type=check_positive, required=True)

    parser.add_argument('--dataset', help='Datsets',
            choices=['arctic', 'blizzard', 'vctk'], required=True)

    parser.add_argument('--lr', help='Initial learning rate',
            type=lib.floatX, default = lib.floatX(0.001))

    parser.add_argument('--grad_clip', help='Upper limit on gradient',
            type=lib.floatX, default = lib.floatX(1.))

    parser.add_argument('--ldim', help='Latent Dimension. O for feedforward mode',
            type=int, default = 0)

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

LEARNING_RATE_DECAY = 2e-4

N_FRAMES = args.n_frames # How many 'frames' to include in each truncated BPTT pass
WEIGHT_NORM = args.weight_norm
SKIP_CONN = args.skip_conn
DIM = args.dim # Model dimensionality.
N_RNN = args.n_rnn # How many RNNs to stack
BIDIRECTIONAL = args.bidirectional
RNN_TYPE = args.rnn_type
H0_MULT = 2 if RNN_TYPE == 'LSTM' else 1
LEARN_H0 = args.learn_h0

if args.ldim == 0:
    LATENT_DIM = None
else:
    LATENT_DIM = args.ldim

BATCH_SIZE = args.batch_size
RESUME = args.resume
EPS = 1e-5
VOCODER_DIM = 63
INPUT_DIM = 420
OUTPUT_DIM = VOCODER_DIM

K_GMM = args.kgmm

DATASET = args.dataset

SPTK_DIR = '/data/lisatmp4/kumarkun/merlin/tools/bin/SPTK-3.9/'
WORLD_DIR = '/data/lisatmp4/kumarkun/merlin/tools/bin/WORLD/'

data_dir = None

for loc in os.environ['FUEL_DATA_PATH'].split(':'):
    if os.path.exists(os.path.join(loc, DATASET)):
        data_dir = loc
        print "Data will be loaded from {}".format(data_dir)

OUT_DIR = os.path.join(FOLDER_PREFIX, tag)

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

def kl_unit_gaussian(mu, log_sigma):
    """
    KL divergence from a unit Gaussian prior
    mean across axis 0 (minibatch), sum across all other axes
    based on yaost, via Alec via Ishaan
    """
    return -0.5 * (1 + 2 * log_sigma - mu**2 - T.exp(2 * log_sigma))


def Encoder(speech, h0, mask):
    """
    Create inference model to infer one single latent variable using bidirectional GRU \
    followed by non-causal dilated convolutions
    """
    if RNN_TYPE == 'GRU':
        rnns_out, last_hidden = lib.ops.stackedGRU('Encoder.GRU',
                                                   N_RNN,
                                                   VOCODER_DIM,
                                                   DIM,
                                                   speech,
                                                   h0=h0,
                                                   weightnorm=WEIGHT_NORM,
                                                   skip_conn=False)
    elif RNN_TYPE == 'LSTM':
        rnns_out, last_hidden = lib.ops.stackedLSTM('Encoder.LSTM',
                                                    N_RNN,
                                                    VOCODER_DIM,
                                                    DIM,
                                                    speech,
                                                    h0=h0,
                                                    weightnorm=WEIGHT_NORM,
                                                    skip_conn=False)

    rnns_out = rnns_out*mask[:,:,None]

    rnns_out = rnns_out.sum(axis = 1)/(mask.sum(axis = 1)[:,None] + lib.floatX(EPS))
    output1 = T.nnet.relu(rnns_out)


    output2 = lib.ops.Linear(
        'Encoder.Output2',
        DIM,
        DIM,
        output1,
        weightnorm=WEIGHT_NORM
    )

    output3 = T.nnet.relu(output2)


    output4 = lib.ops.Linear(
        'Encoder.Output4',
        DIM,
        2*LATENT_DIM,
        output3,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )

    mu = output4[:,::2]
    log_sigma = output4[:,1::2]

    return mu, log_sigma, last_hidden

def Decoder(latent_var, text_features, h0, reset):
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

    if latent_var is not None:
        latent_var_repeated = T.extra_ops.repeat(latent_var[:,None,:], text_features.shape[1], axis = 1)
        features = T.concatenate([text_features, latent_var_repeated], axis = 2)
        RNN_INPUT_DIM = INPUT_DIM + LATENT_DIM
    else:
        RNN_INPUT_DIM = INPUT_DIM
        features = text_features

    if RNN_TYPE == 'LSTM':
        rnns_out, last_hidden = lib.ops.stackedLSTM('Decoder.LSTM',
                                                    N_RNN,
                                                    RNN_INPUT_DIM,
                                                    DIM,
                                                    features,
                                                    h0=h0,
                                                    weightnorm=WEIGHT_NORM,
                                                    skip_conn=SKIP_CONN)
    else:
        rnns_out, last_hidden = lib.ops.stackedGRU('Decoder.GRU',
                                                    N_RNN,
                                                    RNN_INPUT_DIM,
                                                    DIM,
                                                    features,
                                                    h0=h0,
                                                    weightnorm=WEIGHT_NORM,
                                                    skip_conn=SKIP_CONN,
                                                    use_input_every_layer = True)

    output1 = T.nnet.relu(rnns_out)


    output2 = lib.ops.Linear(
        'Decoder.Output1',
        DIM,
        DIM,
        output1,
        weightnorm=WEIGHT_NORM
    )

    output3 = T.nnet.relu(output2)

    output = lib.ops.Linear(
        'Decoder.Output2',
        DIM,
        (2* OUTPUT_DIM + 1)*K_GMM,
        output3,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )


    mu_raw = output[:,:,:OUTPUT_DIM*K_GMM]

    mu = T.clip(mu_raw, lib.floatX(-6.), lib.floatX(6.))

    log_sig = output[:,:,OUTPUT_DIM*K_GMM : 2*OUTPUT_DIM*K_GMM]

    sig = T.exp(log_sig) + lib.floatX(EPS)

    # sig = T.clip(sig, , lib.floatX(2.))

    weights_raw = output[:,:,-K_GMM:]
    weights = T.nnet.softmax(weights_raw.reshape((-1, K_GMM))).reshape(weights_raw.shape) + lib.floatX(EPS)

    return mu, sig, weights, last_hidden




text_features_raw = T.tensor3('text_features_raw') # shape (time-steps, batch, INPUT_DIM)
text_features     = text_features_raw.dimshuffle(1,0,2)

vocoder_audio_raw = T.tensor3('vocoder_audio_raw') # shape (time-steps, batch, VOCODER_DIM)
vocoder_audio     = vocoder_audio_raw.dimshuffle(1,0,2)

h0_enc        = T.tensor3('h0_enc')
h0_dec        = T.tensor3('h0_dec')

reset     = T.iscalar('reset')
lr        = T.scalar('lr')

mask_raw      = T.matrix('mask_raw') # shape (time-steps, batch)
mask = mask_raw.dimshuffle(1,0)

if LATENT_DIM is not None:
    mu_enc, log_sigma_enc, last_hidden_enc = Encoder(vocoder_audio, h0_enc, mask)
    sigma_enc = T.exp(log_sigma_enc) + lib.floatX(EPS)

    eps = T.cast(theano_rng.normal(mu_enc.shape), theano.config.floatX)

    latents = mu_enc + eps*sigma_enc
    kl_cost = kl_unit_gaussian(mu_enc, log_sigma_enc).sum(axis = 1).mean()
else:
    latents = None
    kl_cost = 0.

mu, sigma, weights, last_hidden_dec = Decoder(latents, text_features, h0_dec, reset)

samples = sample_gmm(mu, sigma, weights, theano_rng)

cost_raw = cost_gmm(vocoder_audio, mu, sigma, weights)

cost = T.sum(cost_raw * mask)/(mask.sum() + lib.floatX(EPS)) + kl_cost

params = lib.get_params(cost, lambda x: hasattr(x, 'param') and x.param==True)
lib.print_params_info(params, path=FOLDER_PREFIX)

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

updates = lasagne.updates.adam(grads, params, learning_rate=lr)

if LATENT_DIM is None:
    train_fn = theano.function(
        [text_features_raw, h0_dec, reset, mask_raw, vocoder_audio_raw, lr],
        [cost, last_hidden_dec],
        updates = updates
    )

    valid_fn = theano.function(
        [text_features_raw, h0_dec, reset, mask_raw, vocoder_audio_raw],
        [cost, last_hidden_dec]
    )

    sample_fn = theano.function(
        [text_features_raw, h0_dec, reset],
        [samples, last_hidden_dec]
    )
else:
    train_fn = theano.function(
        [text_features_raw, h0_dec, h0_enc, reset, mask_raw, vocoder_audio_raw, lr],
        [cost, last_hidden_dec, last_hidden_enc],
        updates = updates
    )

    valid_fn = theano.function(
        [text_features_raw, h0_dec, h0_enc, reset, mask_raw, vocoder_audio_raw],
        [cost, last_hidden_dec, last_hidden_enc]
    )

    latents_gen = T.matrix('latent_gen')

    mu_gen, sigma_gen, weights_gen, last_hidden_dec_gen = Decoder(
                                        latents_gen, text_features, h0_dec, reset)
    samples_gen = sample_gmm(mu_gen, sigma_gen, weights_gen, theano_rng)

    sample_fn_temp = theano.function(
        [text_features_raw, latents_gen, h0_dec, reset],
        [samples_gen, last_hidden_dec_gen]
    )

    def sample_fn(text_features_raw_, h0_dec_, reset_):
        latents_generated = numpy.random.normal(
            size= (text_features_raw_.shape[1], LATENT_DIM)
            ).astype(theano.config.floatX)

        return sample_fn_temp(text_features_raw_, latents_generated, h0_dec_, reset_)


h0_init = lib.floatX(numpy.zeros((BATCH_SIZE, N_RNN, H0_MULT*DIM)))

def sampler(save_dir, samples_name, do_post_filtering):
    test_stream = datasets.parrot_stream(
            DATASET,
            use_speaker=False,
            which_sets=('test',),
            batch_size= BATCH_SIZE,
            seq_size=N_FRAMES
    )

    test_iterator = test_stream.get_epoch_iterator()

    last_hidden_dec = h0_init

    actual_so_far_raw, mask_raw, text_features_raw, reset = next(test_iterator)

    samples_so_far, last_hidden_dec = sample_fn(text_features_raw, last_hidden_dec, reset)

    mask_so_far = mask_raw

    actual_so_far_next_raw, mask_raw, text_features_raw, reset = next(test_iterator)

    while reset != 1 :
        samples_next, last_hidden_dec = sample_fn(text_features_raw, last_hidden_dec, reset)

        samples_so_far = numpy.concatenate((samples_so_far, samples_next), axis = 1)
        actual_so_far_raw = numpy.concatenate((actual_so_far_raw, actual_so_far_next_raw), axis = 0)

        actual_so_far_next_raw, mask_raw, text_features_raw, reset = next(test_iterator)

        mask_so_far = numpy.concatenate((mask_so_far, mask_raw), axis = 0)

    actual_so_far = actual_so_far_raw.transpose((1,0,2))


    norm_info_file = os.path.join(
        data_dir, DATASET,
        'norm_info_mgc_lf0_vuv_bap_63_MVN.dat')

    if not os.path.exists(os.path.join(save_dir, 'samples')):
        os.makedirs(os.path.join(save_dir, 'samples'))

    if not os.path.exists(os.path.join(save_dir, 'actual_samples')):
        os.makedirs(os.path.join(save_dir, 'actual_samples'))

    """
    TODO: Remove this commented section.

    """
    for i, this_sample in enumerate(actual_so_far):
        this_sample = this_sample[:int(mask_so_far.sum(axis=0)[i])]

        generate_wav(
            this_sample,
            os.path.join(save_dir, 'actual_samples'),
            samples_name + '_' + str(i),
            sptk_dir = SPTK_DIR,
            world_dir = WORLD_DIR,
            norm_info_file = norm_info_file,
            do_post_filtering = do_post_filtering)
    

    for i, this_sample in enumerate(samples_so_far):
        this_sample = this_sample[:int(mask_so_far.sum(axis=0)[i])]

        generate_wav(
            this_sample,
            os.path.join(save_dir, 'samples'),
            samples_name + '_' + str(i),
            sptk_dir = SPTK_DIR,
            world_dir = WORLD_DIR,
            norm_info_file = norm_info_file,
            do_post_filtering = do_post_filtering)


train_stream = datasets.parrot_stream(
            DATASET,
            use_speaker=False,
            which_sets=('train',),
            batch_size= BATCH_SIZE,
            seq_size=N_FRAMES
        )

valid_stream = datasets.parrot_stream(
            DATASET,
            use_speaker=False,
            which_sets=('valid',),
            batch_size= BATCH_SIZE,
            seq_size=N_FRAMES
        )


total_iters = 0
total_time = 0.

train_costs = []
valid_costs = []

valid_iters = []

print "Training"

for epoch in itertools.count():
    train_iterator = train_stream.get_epoch_iterator()
    valid_iterator = valid_stream.get_epoch_iterator()
    last_hidden_dec = h0_init
    last_hidden_enc = h0_init
    while True:
        try:
            lr_val = lib.floatX(LEARNING_RATE/(1. + LEARNING_RATE_DECAY*total_iters))
            voc_audio_raw, mask_raw, text_features_raw, reset = next(train_iterator)
            
            if LATENT_DIM is None:
                cost, last_hidden_dec =  train_fn(
                            text_features_raw, last_hidden_dec,
                            reset, mask_raw, voc_audio_raw, lr_val
                        )
            else:
                cost, last_hidden_dec, last_hidden_enc =  train_fn(
                            text_features_raw, last_hidden_dec, last_hidden_enc,
                            reset, mask_raw, voc_audio_raw, lr_val
                        )

            train_costs.append(cost)
            total_iters += 1
            if (total_iters % 1000) == 0:
                print "Training NLL at epoch {}, iters {} is {:.4f}".format(
                                epoch, total_iters, numpy.mean(train_costs[-100:])
                            )
        except StopIteration:
            break

    valid_cost_epoch = []
    while True:
        valid_iters.append(total_iters)
        try:
            voc_audio_raw, mask_raw, text_features_raw, reset = next(valid_iterator)

            if LATENT_DIM is None:
                cost, last_hidden_dec =  valid_fn(
                            text_features_raw, last_hidden_dec,
                            reset, mask_raw, voc_audio_raw
                        )
            else:
                cost, last_hidden_dec, last_hidden_enc =  valid_fn(
                            text_features_raw, last_hidden_dec, last_hidden_enc,
                            reset, mask_raw, voc_audio_raw
                        )

            valid_cost_epoch.append(cost)
        except StopIteration:
            val_cost = numpy.mean(valid_cost_epoch)
            print "Validation NLL at epoch {}, iters {} is {:.4f}".format(
                epoch, total_iters, val_cost)
            valid_costs.append(val_cost)
            tag = "epoch_{}_val_score_{:.3f}".format(epoch, val_cost)
            sampler(os.path.join(OUT_DIR, "samples", tag), "sample", False)
            break

