"""
Usage:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -u gan_2.py \
--weight_norm True --skip_conn True --dim 1024 --n_rnn 3 \
--rnn_type GRU --batch_size 8 --dataset vctk --ldim 64
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
from collections import OrderedDict

import datasets
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
    parser.add_argument('--weight_norm', help='Adding learnable weight normalization\
            to all the linear layers (except for the embedding layer)',\
            type=t_or_f, required=True)
    parser.add_argument('--skip_conn', help='Add skip connections to RNN', type=t_or_f, required=True)
    parser.add_argument('--dim', help='Dimension of RNN',\
            type=check_positive, required=True)
    parser.add_argument('--n_rnn', help='Number of layers in the stacked RNN',
            type=check_positive, choices=xrange(1,6), required=True)
    parser.add_argument('--rnn_type', help='GRU or LSTM', choices=['LSTM', 'GRU'],\
            required=True)
    parser.add_argument('--batch_size', help='size of mini-batch',
            type=check_positive, choices=[8, 16, 32, 64, 128, 256], required=True)


    parser.add_argument('--dataset', help='Datsets',
            choices=['arctic', 'blizzard', 'vctk', 'librispeech'], required=True)

    parser.add_argument('--lr', help='Initial learning rate',
            type=lib.floatX, default = lib.floatX(0.001))

    parser.add_argument('--grad_clip', help='Upper limit on gradient',
            type=lib.floatX, default = lib.floatX(1.))

    parser.add_argument('--ldim', help='Latent Dimension. O for feedforward mode',
            type=int, default = 64)

    parser.add_argument('--resume', help='Resume the same model from the last checkpoint.',\
            required=False, default=False, action='store_true')

    parser.add_argument('--exp_name', help='Name of your experiment.',\
            required=False, default="exp")

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

FOLDER_PREFIX = '/Tmp/kumarkun/vocoder_gan/{}'.format(args.exp_name)
GRAD_CLIP = args.grad_clip
LEARNING_RATE = args.lr

LEARNING_RATE_DECAY = 2e-4

WEIGHT_NORM = args.weight_norm
SKIP_CONN = args.skip_conn
DIM = args.dim # Model dimensionality.
N_RNN = args.n_rnn # How many RNNs to stack
RNN_TYPE = args.rnn_type
H0_MULT = 2 if RNN_TYPE == 'LSTM' else 1

NUM_REPEAT = 6

LATENT_DIM = args.ldim

BATCH_SIZE = args.batch_size
RESUME = args.resume
EPS = 1e-6
VOCODER_DIM = 63
INPUT_DIM = 420
OUTPUT_DIM = VOCODER_DIM

DATASET = args.dataset


SPTK_DIR = '/data/lisatmp4/kumarkun/merlin/tools/bin/SPTK-3.9/'
WORLD_DIR = '/data/lisatmp4/kumarkun/merlin/tools/bin/WORLD/'


def floatX(num):
    if theano.config.floatX == 'float32':
        return np.float32(num)
    else:
        raise Exception("{} type not supported".format(theano.config.floatX))

T.nnet.elu = lambda x: T.switch(x >= floatX(0.), x, T.exp(x) - floatX(1.))
T.nnet.lrelu = lambda x: T.switch(x >= floatX(0.), x, 0.001*x)

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


def Encoder(speech, mask, name="", ldim=LATENT_DIM):
    """
    Create inference model to infer one single latent variable using bidirectional GRU \
    followed by non-causal dilated convolutions
    """

    enc_name = "Encoder.{}".format(name)

    learned_h0 = lib.param(
        '{}.h0'.format(enc_name),
        numpy.zeros((N_RNN, H0_MULT*DIM), dtype=theano.config.floatX)
    )
    # Handling LEARN_H0
    learned_h0.param = True
    learned_h0 = T.alloc(learned_h0, speech.shape[0], N_RNN, H0_MULT*DIM)
    learned_h0 = T.unbroadcast(learned_h0, 0, 1, 2)
    h0 = learned_h0

    if RNN_TYPE == 'GRU':
        rnns_out, _ = lib.ops.stackedGRU('{}.GRU'.format(enc_name),
                                                   N_RNN,
                                                   VOCODER_DIM,
                                                   DIM,
                                                   speech,
                                                   h0=h0,
                                                   weightnorm=WEIGHT_NORM,
                                                   skip_conn=False)
    elif RNN_TYPE == 'LSTM':
        rnns_out, _ = lib.ops.stackedLSTM('{}.LSTM'.format(enc_name),
                                                    N_RNN,
                                                    VOCODER_DIM,
                                                    DIM,
                                                    speech,
                                                    h0=h0,
                                                    weightnorm=WEIGHT_NORM,
                                                    skip_conn=False)

    rnns_out = rnns_out*mask[:, :, None]

    rnns_out = rnns_out.sum(axis=1)/(mask.sum(axis=1)[:, None] + lib.floatX(EPS))
    output1 = T.nnet.relu(rnns_out)


    output2 = lib.ops.Linear(
        '{}.Output2'.format(enc_name),
        DIM,
        DIM,
        output1,
        weightnorm=WEIGHT_NORM
    )

    output3 = T.nnet.relu(output2)


    output4 = lib.ops.Linear(
        '{}.Output4'.format(enc_name),
        DIM,
        2*ldim,
        output3,
        initialization='he',
        weightnorm=WEIGHT_NORM
    )

    mu = output4[:,::2]
    log_sigma = output4[:,1::2]

    return mu, log_sigma

def discriminator(predicted, mask, name=""):
    name = "Discriminator.{}.".format(name)
    l1, l2 = Encoder(predicted, mask, name, ldim=1)
    output = (l1+l2)[:,0]
    return output


def Decoder(latent_var, text_features, name=""):
    dec_name = "Decoder.{}".format(name)

    learned_h0 = lib.param(
        '{}.h0'.format(dec_name),
        numpy.zeros((N_RNN, H0_MULT*DIM), dtype=theano.config.floatX)
    )
    # Handling LEARN_H0
    learned_h0.param = True
    learned_h0 = T.alloc(learned_h0, latent_var.shape[0], N_RNN, H0_MULT*DIM)
    learned_h0 = T.unbroadcast(learned_h0, 0, 1, 2)
    h0 = learned_h0
    latent_var_repeated = T.extra_ops.repeat(latent_var[:, None, :], text_features.shape[1], axis=1)
    features = T.concatenate([text_features, latent_var_repeated], axis=2)
    RNN_INPUT_DIM = INPUT_DIM + LATENT_DIM

    if RNN_TYPE == 'LSTM':
        rnns_out, last_hidden = lib.ops.stackedLSTM('{}.LSTM'.format(dec_name),
                                                    N_RNN,
                                                    RNN_INPUT_DIM,
                                                    DIM,
                                                    features,
                                                    h0=h0,
                                                    weightnorm=WEIGHT_NORM,
                                                    skip_conn=SKIP_CONN)
    else:
        rnns_out, last_hidden = lib.ops.stackedGRU('{}.GRU'.format(dec_name),
                                                    N_RNN,
                                                    RNN_INPUT_DIM,
                                                    DIM,
                                                    features,
                                                    h0=h0,
                                                    weightnorm=WEIGHT_NORM,
                                                    skip_conn=SKIP_CONN,
                                                    use_input_every_layer=True)

    output1 = T.nnet.relu(rnns_out)


    output2 = lib.ops.Linear(
        '{}.Output1'.format(dec_name),
        DIM,
        DIM,
        output1,
        weightnorm=WEIGHT_NORM
    )

    output3 = T.nnet.relu(output2)

    output = lib.ops.Linear(
            '{}.Output2'.format(dec_name),
            DIM,
            OUTPUT_DIM,
            output3,
            initialization='he',
            weightnorm=WEIGHT_NORM
    )
    return output


text_features_raw = T.tensor3('text_features_raw') # shape (time-steps, batch, INPUT_DIM)
text_features     = text_features_raw.dimshuffle(1,0,2)

vocoder_audio_raw = T.tensor3('vocoder_audio_raw') # shape (time-steps, batch, VOCODER_DIM)
vocoder_audio     = vocoder_audio_raw.dimshuffle(1,0,2)

lr_gen        = T.scalar('lr_gen')
lr_desc       = T.scalar('lr_desc')

fakeness  = T.scalar('fakeness')


mask_raw  = T.matrix('mask_raw') # shape (time-steps, batch)
mask      = mask_raw.dimshuffle(1,0)


mu_enc, log_sigma_enc= Encoder(vocoder_audio, mask)
sigma_enc = T.exp(log_sigma_enc) + lib.floatX(EPS)

eps = T.cast(theano_rng.normal(mu_enc.shape), theano.config.floatX)

latents = mu_enc + eps*sigma_enc
kl_cost = kl_unit_gaussian(mu_enc, log_sigma_enc).sum(axis=1).mean()


output = Decoder(latents, text_features)

disc_out_gen = discriminator(output, mask)
disc_out_data = discriminator(vocoder_audio, mask)

disc_cost = (T.nnet.softplus(disc_out_gen) + T.nnet.softplus(-disc_out_data)).mean()
gen_cost =  (T.nnet.softplus(-disc_out_gen) + T.nnet.softplus(disc_out_data)).mean()

samples = output
cost_raw = T.sum((samples - vocoder_audio) ** 2, axis=-1)

kl_cost = kl_cost/(mask.sum() + lib.floatX(EPS))
reconst_cost = T.sum(cost_raw * mask)/(mask.sum() + lib.floatX(EPS))

cost = kl_cost + 0.5*reconst_cost

discriminator_cost = disc_cost
generator_cost = cost + gen_cost

discriminator_params = lib.get_params(discriminator_cost, lambda x: (hasattr(x, 'param') and x.param==True) and ("Discriminator" in x.name))
generator_params = lib.get_params(generator_cost, lambda x: (hasattr(x, 'param') and x.param==True) and ("Discriminator" not in x.name))

params = discriminator_params + generator_params

lib.print_params_info(params, path=FOLDER_PREFIX)

discriminator_grads = T.grad(discriminator_cost, wrt=discriminator_params, disconnected_inputs='warn')
generator_grads = T.grad(generator_cost, wrt=generator_params, disconnected_inputs='warn')

discriminator_grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in discriminator_grads]
generator_grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in generator_grads]


discriminator_updates = lasagne.updates.sgd(discriminator_grads, discriminator_params, learning_rate=lr_desc)
generator_updates = lasagne.updates.adam(generator_grads, generator_params, learning_rate=lr_gen)

updates = OrderedDict()
for key in discriminator_updates.keys():
    updates[key] = discriminator_updates[key]

for key in generator_updates.keys():
    updates[key] = generator_updates[key]


# train_disc = theano.function(
#         [text_features_raw, mask_raw, vocoder_audio_raw, lr_desc],
#         [discriminator_cost],
#         updates=discriminator_updates
# )
# train_generator = theano.function(
#     [text_features_raw, mask_raw, vocoder_audio_raw, lr_gen],
#     [cost, reconst_cost, kl_cost, generator_cost],
#     updates=generator_updates
# )

train_both = theano.function(
        [text_features_raw, mask_raw, vocoder_audio_raw, lr_desc, lr_gen],
        [cost, reconst_cost, kl_cost, discriminator_cost, generator_cost],
        updates=updates
)

train_output = ['cost', 'reconst_cost', 'kl_cost', 'discriminator_cost', 'generator_cost']

valid_fn = theano.function(
    [text_features_raw, mask_raw, vocoder_audio_raw],
    [reconst_cost, kl_cost]
)

latents_gen = T.matrix('latent_gen')

output_gen = Decoder(latents_gen, text_features)

samples_gen = output_gen

sample_fn = theano.function(
    [text_features_raw, latents_gen],
    [samples_gen]
)

def sampler(save_dir, samples_name, do_post_filtering):
    test_stream = datasets.parrot_stream(
            DATASET,
            use_speaker=False,
            which_sets=('test',),
            batch_size= BATCH_SIZE,
            seq_size=10000
    )

    test_iterator = test_stream.get_epoch_iterator()

    latents_generated = numpy.random.normal(size=(NUM_REPEAT, LATENT_DIM))

    latents_generated = lib.floatX(numpy.tile(latents_generated, (BATCH_SIZE, 1)))

    actual_so_far_raw, mask_raw, text_features_raw, reset = next(test_iterator)

    text_features_raw_repeated = numpy.repeat(text_features_raw, NUM_REPEAT, axis = 1)


    samples_so_far = sample_fn(text_features_raw_repeated, latents_generated)

    mask_so_far = mask_raw

    actual_so_far = actual_so_far_raw.transpose((1,0,2))

    mask_so_far_repeated = numpy.repeat(mask_so_far, NUM_REPEAT, axis = 1)

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
            sptk_dir=SPTK_DIR,
            world_dir=WORLD_DIR,
            norm_info_file=norm_info_file,
            do_post_filtering=do_post_filtering)
    

    for i, this_sample in enumerate(samples_so_far):
        this_sample = this_sample[:int(mask_so_far_repeated.sum(axis=0)[i])]

        generate_wav(
            this_sample,
            os.path.join(save_dir, 'samples'),
            samples_name + '_' + str(i//NUM_REPEAT) + '_latent_' + str(i % NUM_REPEAT),
            sptk_dir=SPTK_DIR,
            world_dir=WORLD_DIR,
            norm_info_file=norm_info_file,
            do_post_filtering=do_post_filtering)

sampler(os.path.join(OUT_DIR, "samples", "initial_samples"), "sample", False)


train_stream = datasets.parrot_stream(
            DATASET,
            use_speaker=False,
            which_sets=('train',),
            batch_size=BATCH_SIZE,
            seq_size=10000
        )

valid_stream = datasets.parrot_stream(
            DATASET,
            use_speaker=False,
            which_sets=('valid',),
            batch_size= BATCH_SIZE,
            seq_size=10000
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
    KL_costs_train_epoch = []
    KL_costs_valid_epoch = []
    generator_cost_epoch = []
    discriminator_cost_epoch = []

    train_stats = [[] for i in train_output]

    while True:
        try:
            lr_val = lib.floatX(LEARNING_RATE/(1. + LEARNING_RATE_DECAY*total_iters))
            voc_audio_raw, mask_raw, text_features_raw, reset = next(train_iterator)
            assert(reset == 1.)
            
            train_res = train_both(text_features_raw, mask_raw, voc_audio_raw, lr_val, lib.floatX(0.1*lr_val))

            for i, stat in enumerate(train_output):
                train_stats[i].append(train_res[i])

            # print(train_stats)

            total_iters += 1
            if (total_iters % 100) == 0:
                print "Train iterations : {}".format(total_iters)
                for i, stat in enumerate(train_output):
                    print "\t{}: {}".format(stat, np.mean(train_stats[i]))
                    train_stats[i] = []
        except StopIteration:
            break

    valid_cost_epoch = []
    while True:
        valid_iters.append(total_iters)
        try:
            voc_audio_raw, mask_raw, text_features_raw, reset = next(valid_iterator)

            reconst, kl_cost = valid_fn(text_features_raw, mask_raw, voc_audio_raw)

            KL_costs_valid_epoch.append(kl_cost)

            valid_cost_epoch.append(reconst)
        except StopIteration:
            val_cost = numpy.mean(valid_cost_epoch)
            print "Reconstruction cost at epoch {}, iters {} is {:.4f}".format(
                epoch, total_iters, val_cost)

            if len(KL_costs_valid_epoch) > 0:
                print "KL cost is {:.4f}".format(numpy.mean(KL_costs_valid_epoch))
            valid_costs.append(val_cost)
            tag = "epoch_{}_val_score_{:.3f}".format(epoch, val_cost)
            sampler(os.path.join(OUT_DIR, "samples", tag), "sample", False)
            break
