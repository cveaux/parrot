"""
Usage: python vocoder_to_raw_wavenet.py --exp v2r --dim 128 --q_levels 256 --q_type mu-law --batch_size 8 \
--wavenet_blocks 4 --dilation_layers_per_block 6 --sequence_len_to_train 1600
"""



import sys
import os

assert(os.environ['NN_LIB'])
sys.path.insert(1, os.environ['NN_LIB'])
import argparse

import lib
import lib.ops
import numpy
import theano

import theano.tensor as T
import datasets
from quantize import __batch_quantize, mu2linear
import lasagne
import scipy.io
import scipy.io.wavfile
import time

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
    parser = argparse.ArgumentParser(description='Vocoder to raw')
    parser.add_argument('--exp', help='Experiment name',
            type=str, required=False, default='v2r')
    parser.add_argument('--dim', help='Dimension of RNN and MLPs',\
            type=check_positive, required=True)
    parser.add_argument('--q_levels', help='Number of bins for quantization of audio samples. Should be 256 for mu-law.',\
            type=check_positive, required=True)
    parser.add_argument('--q_type', help='Quantization in linear-scale, a-law-companding, or mu-law compandig. With mu-/a-law quantization level shoud be set as 256',\
            choices=['linear', 'a-law', 'mu-law'], required=True)
    parser.add_argument('--batch_size', help='size of mini-batch',
            type=check_positive, choices=[8, 16, 32, 64, 128, 256], required=True)
    parser.add_argument('--wavenet_blocks', help='Number of wavnet blocks to use',
            type=check_positive, required=True)
    parser.add_argument('--dilation_layers_per_block', help='number of dilation layers per block',
            type=check_positive, required=True)

    parser.add_argument('--sequence_len_to_train', help='size of output map',
            type=check_positive, required=True)

    args = parser.parse_args()
    tag = args.exp

    return args, tag

args, tag = get_args()

OVERLAP = (2**args.dilation_layers_per_block - 1)*args.wavenet_blocks + 1
DIM = args.dim # Model dimensionality.
Q_LEVELS = args.q_levels # How many levels to use when discretizing samples. e.g. 256 = 8-bit scalar quantization
Q_TYPE = args.q_type # log- or linear-scale

if Q_TYPE == 'mu-law' and Q_LEVELS != 256:
    raise ValueError('For mu-law Quantization levels should be exactly 256!')

if Q_TYPE == 'mu-law' and Q_LEVELS != 256:
    raise ValueError('For mu-law Quantization levels should be exactly 256!')

# Fixed hyperparams
GRAD_CLIP = 1 # Elementwise grad clip threshold
BITRATE = 16000

FOLDER_PREFIX = os.path.join('/Tmp/kumarkun/vocoder_to_raw_wavenets/results', tag)
SEQ_LEN = args.sequence_len_to_train # Total length (# of samples) in one batch
Q_ZERO = numpy.int32(Q_LEVELS//2) # Discrete value correponding to zero amplitude

assert((SEQ_LEN % 80) == 0)

VOCODER_FRAME_NUM = SEQ_LEN//80

LEARNING_RATE = lib.floatX(numpy.float32(0.0001))

if not os.path.exists(FOLDER_PREFIX):
    os.makedirs(FOLDER_PREFIX)

PARAMS_PATH = os.path.join(FOLDER_PREFIX, 'params')

if not os.path.exists(PARAMS_PATH):
    os.makedirs(PARAMS_PATH)

SAMPLES_PATH = os.path.join(FOLDER_PREFIX, 'samples_generated')
ORIGINAL_SAMPLES_PATH = os.path.join(FOLDER_PREFIX, 'original_test_samples')

if not os.path.exists(SAMPLES_PATH):
    os.makedirs(SAMPLES_PATH)

if not os.path.exists(ORIGINAL_SAMPLES_PATH):
    os.makedirs(ORIGINAL_SAMPLES_PATH)

BEST_PATH = os.path.join(FOLDER_PREFIX, 'best')

if not os.path.exists(BEST_PATH):
    os.makedirs(BEST_PATH)

lib.print_model_settings(locals(), path=FOLDER_PREFIX, sys_arg=False)

def create_wavenet_block(inp, num_dilation_layer, input_dim, output_dim, constant_value_to_add=None, name=None):
    assert name is not None
    layer_out = inp
    skip_contrib = []
    for i in range(num_dilation_layer):
        layer_out, skip_c = lib.ops.dil_conv_1D(
                    layer_out,
                    output_dim,
                    input_dim if i == 0 else output_dim,
                    2,
                    dilation=2**i,
                    non_linearity='gated',
                    name=name+".dilation_{}".format(i+1)
                )
        if constant_value_to_add is not None:
            layer_out = T.inc_subtensor(layer_out[:, -constant_value_to_add.shape[1]:], constant_value_to_add)

        skip_contrib.append(skip_c)

    skip_out = skip_contrib[-1]

    j = 0
    for i in range(num_dilation_layer-1):
        j += 2**(num_dilation_layer-i-1)
        skip_out = skip_out + skip_contrib[num_dilation_layer-2 - i][:,j:]

    return layer_out, skip_out

def create_model(inp, extra_block_constants=None):
    out = (inp.astype(theano.config.floatX)/lib.floatX(Q_LEVELS-1) - lib.floatX(0.5))
    l_out = out.dimshuffle(0,1,'x')

    skips = []
    for i in range(args.wavenet_blocks):
        if extra_block_constants is not None:
            curr_block_constant = extra_block_constants[i]
        else:
            curr_block_constant = None

        l_out, skip_out = create_wavenet_block(l_out, args.dilation_layers_per_block, 1 if i == 0 else args.dim, args.dim, curr_block_constant, name = "block_{}".format(i+1))
        skips.append(skip_out)

    out = skips[-1]

    for i in range(args.wavenet_blocks -  1):
        out = out + skips[args.wavenet_blocks - 2 - i][:, (2**args.dilation_layers_per_block - 1)*(i+1):]

    if extra_block_constants is not None:
        out = out + extra_block_constants[-1]

    for i in range(3):
        out = lib.ops.conv1d("out_{}".format(i+1), out, args.dim, args.dim, 1, non_linearity='relu')

    out = lib.ops.conv1d("final", out, args.dim, args.q_levels, 1, non_linearity='identity')

    return out

sequences = T.imatrix('sequences')
input_seq = sequences[:,:-1]

target_sequences = sequences[:,OVERLAP:]
vocoder_frame = T.tensor3('vocoder_frame')
vocoder_mask = T.matrix('vocoder_mask')
target_mask = T.extra_ops.repeat(vocoder_mask, 80, axis=1)

vocoder_to_wavenets = lib.ops.Linear("vocoder_to_wavenet", 63, (args.wavenet_blocks + 1)*DIM, vocoder_frame)
vocoder_to_wavenets_raw_resolution = T.extra_ops.repeat(vocoder_to_wavenets, 80, axis=1)

vocoder_to_blocks = []

for i in range(args.wavenet_blocks + 1):
    vocoder_to_blocks.append(vocoder_to_wavenets_raw_resolution[:, :, i*DIM:(i+1)*DIM])


output = create_model(input_seq, vocoder_to_blocks)

cost = T.nnet.categorical_crossentropy(
    T.nnet.softmax(output.reshape((-1, Q_LEVELS))),
    target_sequences.flatten()
)
cost = cost.reshape(target_sequences.shape)
cost = (cost * target_mask).sum()/(target_mask.sum() + 1e-5)
cost = cost * lib.floatX(numpy.log2(numpy.e))

params = lib.get_params(cost, lambda x: hasattr(x, 'param') and x.param==True)
lib.print_params_info(params, path=FOLDER_PREFIX)

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]

updates = lasagne.updates.adam(grads, params, learning_rate=LEARNING_RATE)

print "Compiling functions"
# Training function
train_fn = theano.function(
    [sequences, vocoder_frame, vocoder_mask],
    cost,
    updates=updates,
    on_unused_input='warn'
)

# Validation and Test function
# test_fn = theano.function(
#     [sequences, vocoder_frame, vocoder_mask],
#     cost,
#     on_unused_input='warn'
# )

# Sampling at frame level
generate_fn = theano.function(
    [sequences, vocoder_frame],
    lib.ops.softmax_and_sample(output),
    on_unused_input='warn'
)


def generate_and_save_samples(tag):
    def write_audio_file(name, data):
        data = data.astype('float32')
        data -= data.min()
        data /= data.max()
        data -= 0.5
        data *= 0.95
        scipy.io.wavfile.write(
                    os.path.join(SAMPLES_PATH, name+'.wav'),
                    BITRATE,
                    data)

    total_time = time.time()
    # Generate N_SEQS' sample files, each 5 seconds long

    test_data_stream = datasets.parrot_stream('dimex', labels_type='text', which_sets=('test',),
             seq_size=10000, batch_size=8, raw_audio=True)
    ep_iter = test_data_stream.get_epoch_iterator()

    data_batch = next(ep_iter)
    voc_frame = data_batch[0].transpose(1, 0, 2)
    voc_mask = data_batch[1].transpose(1, 0)
    raw_data = data_batch[2]
    # print voc_frame.shape

    LENGTH = voc_frame.shape[1]*80


    num_prev_samples_to_use = OVERLAP
    N_SEQS = 8

    samples = numpy.zeros((N_SEQS, LENGTH + num_prev_samples_to_use+161), dtype='int32')
    samples[:, :num_prev_samples_to_use] = Q_ZERO

    for t1 in range(voc_frame.shape[1]):
        for t in range(80):
            samples[:,num_prev_samples_to_use+t1*80+t] = generate_fn(samples[:, t1*80+t:t +(t1+1)*80+ num_prev_samples_to_use], voc_frame[:,t1:t1+1])[:,0]


    total_time = time.time() - total_time
    log = "{} samples of {} seconds length generated in {} seconds."
    log = log.format(N_SEQS, LENGTH/16000.0, total_time)
    print log

    lib.save_params(os.path.join(PARAMS_PATH, tag+".pkl"))

    for i in xrange(N_SEQS):
        samp = samples[i, num_prev_samples_to_use: voc_mask[i].sum()*80+num_prev_samples_to_use]
        if Q_TYPE == 'mu-law':
            samp = mu2linear(samp)
        elif Q_TYPE == 'a-law':
            raise NotImplementedError('a-law is not implemented')
        write_audio_file("sample_{}_{}".format(tag, i), samp)
        if tag == "initial_samples":
            scipy.io.wavfile.write(
                    os.path.join(ORIGINAL_SAMPLES_PATH, 'original_{}.wav'.format(i)),
                    BITRATE,
                    raw_data[i][:voc_mask[i].sum()*80])

generate_and_save_samples("initial_samples")

data_stream = datasets.parrot_stream('dimex', labels_type='text', seq_size=10000, batch_size=8, raw_audio=True)
total_train_time = 0.
last_print_time = 0.
costs = []
iters = 0
for ep in range(100):
    ep_iter = data_stream.get_epoch_iterator()
    while(True):
        start_time = time.time()
        try:
            data_batch = next(ep_iter)
            voc_frame = data_batch[0].transpose(1, 0, 2)
            voc_mask = data_batch[1].transpose(1, 0)
            raw_data = __batch_quantize(data_batch[2], Q_LEVELS, Q_TYPE)
            zeros_to_append_raw = numpy.zeros((8, OVERLAP), dtype=numpy.int32) + Q_ZERO
            appended_raw_data = numpy.concatenate([zeros_to_append_raw, raw_data], axis=1)
            for i in range(voc_frame.shape[1]//VOCODER_FRAME_NUM):
                curr_voc_frame = voc_frame[:, i*VOCODER_FRAME_NUM:(i+1)*VOCODER_FRAME_NUM]
                curr_voc_mask  = voc_mask[:, i*VOCODER_FRAME_NUM:(i+1)*VOCODER_FRAME_NUM]
                curr_raw_data = appended_raw_data[:, i*VOCODER_FRAME_NUM*80: (i+1)*VOCODER_FRAME_NUM*80 + OVERLAP]
                c_ = train_fn(curr_raw_data, curr_voc_frame, curr_voc_mask)
                costs.append(c_)
                iters += 1
        except StopIteration:
            break
        total_train_time += time.time() - start_time
        if total_train_time - last_print_time > 5400.:
            print("Train cost after training {} hrs at epoch {}, iters {}: {}".format(
            											total_train_time/3600., ep, iters, numpy.mean(costs)))
            last_print_time = total_train_time
            costs = []
            generate_and_save_samples('iters_{}_ep_{}'.format(iters, ep))
