"""Sampling code for the parrot.

Usage: THEANO_FLAGS="mode=FAST_RUN,floatX=float32,device=gpu,lib.cnmem=.95" python -u sample.py --experiment_name bidirectional_encoder --dataset vctk --samples_name dummy

Loads the trained model and samples.
"""

import numpy
import os
import cPickle
import cPickle as pickle
import logging

from blocks.serialization import load_parameters
from blocks.model import Model
from blocks.graph import ComputationGraph

from datasets import parrot_stream, get_quantizers
from model import Parrot
from utils import (
    sample_parse,  create_animation, numpy_one_hot, write_audio_file)
from generate import generate_wav

from quantize import mu2linear

import theano

from blocks.filter import VariableFilter
from blocks.roles import PARAMETER



floatX = theano.config.floatX

logging.basicConfig()

data_dir = os.environ['FUEL_DATA_PATH']
args = sample_parse()


with open(os.path.join(
        args.save_dir, 'config',
        args.experiment_name + '.pkl')) as f:
    saved_args = cPickle.load(f)

# print saved_args
# Set default values for old config files.
if not hasattr(saved_args, 'weak_feedback'):
    saved_args.weak_feedback = False
if not hasattr(saved_args, 'full_feedback'):
    saved_args.full_feedback = False
if not hasattr(saved_args, 'very_weak_feedback'):
    saved_args.very_weak_feedback = False
if not hasattr(saved_args, 'labels_type'):
    saved_args.labels_type = 'full'
if not hasattr(saved_args, 'quantized_input'):
    saved_args.quantized_input = False
if not hasattr(saved_args, 'use_latent'):
    saved_args.use_latent = False
if not hasattr(saved_args, 'latent_dim'):
    saved_args.latent_dim = 64
if not hasattr(saved_args, 'raw_output'):
    saved_args.raw_output = False

if not os.path.exists(os.path.join(args.save_dir, "samples")):
    os.makedirs(os.path.join(args.save_dir, "samples"))

assert saved_args.dataset == args.dataset

if args.use_last:
    params_mode = 'last_'
else:
    params_mode = 'best_'

args.samples_name = params_mode + args.samples_name

with open(os.path.join(
        args.save_dir, "pkl",
        params_mode + args.experiment_name + ".tar"), 'rb') as src:
    parameters = load_parameters(src)


parrot_args = {
    'input_dim': saved_args.input_dim,
    'output_dim': saved_args.output_dim,
    'rnn_h_dim': saved_args.rnn_h_dim,
    'readouts_dim': saved_args.readouts_dim,
    'labels_type': saved_args.labels_type,
    'weak_feedback': saved_args.weak_feedback,
    'full_feedback': saved_args.full_feedback,
    'very_weak_feedback': saved_args.very_weak_feedback,
    'feedback_noise_level': None,
    'only_noise': False,
    'layer_norm': saved_args.layer_norm,
    'use_speaker': saved_args.use_speaker,
    'num_speakers': saved_args.num_speakers,
    'speaker_dim': saved_args.speaker_dim,
    'which_cost': saved_args.which_cost,
    'num_characters': saved_args.num_characters,
    'attention_type': saved_args.attention_type,
    'attention_alignment': saved_args.attention_alignment,
    'sampling_bias': args.sampling_bias,
    'sharpening_coeff': args.sharpening_coeff,
    'timing_coeff': args.timing_coeff,
    'encoder_type': saved_args.encoder_type,
    'only_residual_train': args.only_residual_train,
    'quantized_input': args.quantized_input,
    'use_latent': saved_args.use_latent,
    'latent_dim': saved_args.latent_dim,
    'raw_output': saved_args.raw_output,
    'optimise_for_a_single_speaker':True,
    'name': 'parrot'}

# print parrot_args

parrot = Parrot(**parrot_args)

if saved_args.use_latent:
    LATENT_NUM = 5
    LATENT_DIM = parrot.latent_dim
else:
    LATENT_NUM = 1
    LATENT_DIM = parrot.latent_dim

features, features_mask, labels, labels_mask, speaker, latent_var, start_flag, raw_audio = \
    parrot.symbolic_input_variables()

cost, extra_updates, attention_vars, kl_cost, mutual_info = parrot.compute_cost(
    features, features_mask, labels, labels_mask,
    speaker, start_flag, args.num_samples, raw_audio=raw_audio)


cg = ComputationGraph(cost)
model = Model(cost)
model.set_parameter_values(parameters)
parameters = cg.parameters

var_filter = VariableFilter(roles=[PARAMETER], bricks=[parrot.new_speaker_emb])
parameters_to_tune = var_filter(parameters)

print "Successfully loaded the parameters."



norm_info_file = os.path.join(
    data_dir, args.dataset,
    'norm_info_mgc_lf0_vuv_bap_63_MVN.dat')


# import ipdb; ipdb.set_trace()

import lasagne
import theano.tensor as tensor

lr = tensor.scalar('lr')
updates = lasagne.updates.sgd(cost, parameters_to_tune, learning_rate=lr)

# print len(parameters_to_tune)

parameter_val = parameters_to_tune[0].get_value()
# print parameter_val

parameters_to_tune[0].set_value(numpy.random.uniform(low=-0.2, high=0.2, size=parameter_val.shape).astype('float32'))

# updates = updates + extra_updates

print("Compiling theano finetune function")

embed_learn_fn = theano.function([features, features_mask, labels, labels_mask, speaker, start_flag, lr], cost, updates=updates)


if args.new_sentences:
    numpy.random.seed(1)
    speaker_tr = numpy.random.random_integers(
        1, saved_args.num_speakers - 1, (args.num_samples, 1))
    speaker_tr = numpy.int8(speaker_tr)

    new_sentences_file = os.path.join(
        data_dir, args.dataset,
        'new_sentences.npy')

    print 'Loading sentences from : ' + new_sentences_file

    labels_tr = numpy.load(new_sentences_file)
    lengths_tr = [len(x) for x in labels_tr]
    max_length = max(lengths_tr)
    features_mask_tr = numpy.zeros(
        (args.num_samples, max_length), dtype='float32')
    padded_labels_tr = numpy.zeros(
        (args.num_samples, max_length, saved_args.input_dim), dtype='float32')

    for i, sample in enumerate(labels_tr):
        padded_labels_tr[i, :len(sample)] = sample
        features_mask_tr[i, :len(sample)] = 1.

    labels_tr = padded_labels_tr

    features_mask_tr = features_mask_tr.swapaxes(0, 1)
    labels_tr = labels_tr.swapaxes(0, 1)
else:

    test_stream = parrot_stream(
        args.new_dataset, False, ('test',), args.num_samples,
        10000, sorting_mult=1, labels_type=saved_args.labels_type,
        quantize_features=saved_args.quantized_input)

    data_tr = next(test_stream.get_epoch_iterator())
    data_tr = {
        source: data for source, data in zip(test_stream.sources, data_tr)}

    print "Loaded sources from test_stream: ", data_tr.keys()
    features_tr = data_tr.get('features', None)
    features_mask_tr = data_tr.get('features_mask', None)
    speaker_tr = data_tr.get('speaker_index', None)
    labels_tr = data_tr.get('labels', None)
    labels_mask_tr = data_tr.get('labels_mask', None)
    start_flag_tr = data_tr.get('start_flag', None)

    raw_audio_tr = data_tr.get('raw_audio', None)

args.random_speaker = True

if args.random_speaker:
    numpy.random.seed(1)
    speaker_tr = numpy.random.random_integers(
        1, saved_args.num_speakers - 1, (args.num_samples, 1))
    speaker_tr = numpy.int8(speaker_tr)

if args.phrase is not None:
    import pickle
    data_path = os.environ['FUEL_DATA_PATH']
    char2code_path = os.path.join(data_path, args.new_dataset, 'char2code.pkl')
    with open(char2code_path, 'r') as f:
        char2code = pickle.load(f)
    labels_tr = numpy.array([char2code[x] for x in args.phrase], dtype='int8')
    labels_tr = numpy.tile(labels_tr, (args.num_samples, 1))
    labels_mask_tr = numpy.ones(labels_tr.shape, dtype='float32')

if args.speaker_id and saved_args.use_speaker:
    speaker_tr = speaker_tr * 0 + args.speaker_id


if args.mix and saved_args.use_speaker:
    speaker_tr = speaker_tr * 0
    parameters['/parrot/lookuptable.W'][0] = \
        args.mix * parameters['/parrot/lookuptable.W'][10] + \
        (1 - args.mix) * parameters['/parrot/lookuptable.W'][11]


def label_code_transform(labels_tr):
    label_code_conversion = pickle.load(open(os.path.join(data_dir,
                     "{}2{}_phone_code_conversion.pkl".format(args.new_dataset, args.dataset)), "rb"))
    label_code_conversion[0] = 0

    labels_tr_shape = labels_tr.shape
    labels_tr_flattened = labels_tr.flatten()

    for i in range(len(labels_tr_flattened)):
        labels_tr_flattened[i] = label_code_conversion[labels_tr_flattened[i]]

    labels_tr = labels_tr_flattened.reshape(labels_tr_shape)
    return labels_tr


if not os.path.exists(os.path.join(args.save_dir, 'samples', 'adaptation')):
    os.makedirs(os.path.join(args.save_dir, 'samples', 'adaptation'))

if args.sample_one_step:
    gen_x, gen_k, gen_w, gen_pi, gen_phi, gen_pi_att = \
        parrot.sample_using_input(data_tr, args.num_samples)
else:
    labels_tr = label_code_transform(labels_tr)

    new_speakers_tr = numpy.arange(args.num_samples).reshape((args.num_samples, 1)).astype('int32')

    costs = []
    data_tr = next(test_stream.get_epoch_iterator())
    data_tr = {
        source: data for source, data in zip(test_stream.sources, data_tr)}

    print "Loaded sources from test_stream: ", data_tr.keys()
    features_tr_old = data_tr.get('features', None)
    features_mask_tr_old = data_tr.get('features_mask', None)
    speaker_tr_old = data_tr.get('speaker_index', None)
    labels_tr_old = data_tr.get('labels', None)
    labels_mask_tr_old = data_tr.get('labels_mask', None)
    start_flag_tr_old = data_tr.get('start_flag', None)

    raw_audio_tr_old = data_tr.get('raw_audio', None)


    labels_tr_old = numpy.repeat(labels_tr_old, LATENT_NUM, axis=0)
    labels_mask_tr_old = numpy.repeat(labels_mask_tr_old, LATENT_NUM, axis=0)
    features_mask_tr_old = numpy.repeat(features_mask_tr_old, LATENT_NUM, axis=1)
    new_speakers_tr_old = numpy.repeat(new_speakers_tr, LATENT_NUM, axis=0)

    latent_var_tr_old = numpy.random.normal(size=(LATENT_NUM, LATENT_DIM)).astype(floatX)
    latent_var_tr_old = numpy.tile(latent_var_tr_old, (args.num_samples, 1))


    for i in range(10000):
        lr_ =  numpy.float32(0.001/(1+ 3*(i // 1000)))
        cost_ = embed_learn_fn(features_tr, features_mask_tr, labels_tr, labels_mask_tr, new_speakers_tr, 1., lr_)
        costs.append(cost_)
        if (i % 200) == 0:
            # import ipdb; ipdb.set_trace()
            # test_stream = parrot_stream(
            #     args.new_dataset, False, ('test',), args.num_samples,
            #     10000, sorting_mult=1, labels_type=saved_args.labels_type,
            #     quantize_features=saved_args.quantized_input)

            gen_x, gen_k, gen_w, gen_pi, gen_phi, gen_pi_att = parrot.sample_model(
                labels_tr_old, labels_mask_tr_old, features_mask_tr_old, 
                new_speakers_tr_old, latent_var_tr_old, LATENT_NUM * args.num_samples)

            for j, this_sample in enumerate(gen_x.swapaxes(1, 0)):
                this_sample = this_sample[:int(features_mask_tr_old.sum(axis=0)[j])]
                generate_wav(
                    this_sample,
                    os.path.join(args.save_dir, 'samples', 'adaptation'),
                    "sample_{}_{}_iters_{}".format(args.samples_name, j, i),
                    sptk_dir=args.sptk_dir,
                    world_dir=args.world_dir,
                    norm_info_file=norm_info_file,
                    do_post_filtering=args.do_post_filtering)
            if len(costs) != 0:
                print "cost at iter {} is {}".format(i, numpy.mean(costs[-200:]))

print "Successfully sampled the parrot."

if saved_args.quantized_input:
    _, dequantize = get_quantizers()
    gen_x = dequantize(gen_x)

# import ipdb; ipdb.set_trace()

if args.save_new:
    if not os.path.exists(os.path.join(args.save_dir, 'original_samples', args.new_dataset)):
        os.makedirs(os.path.join(args.save_dir, 'original_samples', args.new_dataset))

    for i, this_sample in enumerate(features_tr.swapaxes(1, 0)):
        this_sample = this_sample[:int(features_mask_tr.sum(axis=0)[i])]
        generate_wav(
            this_sample,
            os.path.join(args.save_dir, 'original_samples', args.new_dataset),
            "sample_{}".format(i),
            sptk_dir=args.sptk_dir,
            world_dir=args.world_dir,
            norm_info_file=norm_info_file,
            do_post_filtering=args.do_post_filtering)



if saved_args.raw_output:
    print "Saving original raw audio"

    to_save_path = os.path.join(args.save_dir, 'samples', 'original_raw')
    if not os.path.exists(to_save_path):
        os.makedirs(to_save_path)
    
    raw_audio = raw_audio_tr.transpose(1, 0, 2).reshape((raw_audio_tr.shape[1], -1))
    for i in range(len(raw_audio)):
        write_audio_file("original_raw_{}.wav".format(i), mu2linear(raw_audio[i][:int(features_mask_tr.sum(axis=0)[i])*80]), to_save_path)

    print "Sampling and saving raw audio..."

    to_save_path = os.path.join(args.save_dir, 'samples', 'raw')
    if not os.path.exists(to_save_path):
        os.makedirs(to_save_path)

    parrot.sampleRnn.sample_raw(gen_x.copy(), None, args.samples_name, to_save_path)
    print "Successfully sampled raw audio..."

gen_x = gen_x.swapaxes(0, 1)
# import ipdb; ipdb.set_trace()

if saved_args.labels_type in ['unaligned_phonemes', 'text']:
    from utils import full_plot
    gen_k = gen_k.swapaxes(0, 1)
    gen_w = gen_w.swapaxes(0, 1)
    gen_pi = gen_pi.swapaxes(0, 1)
    gen_phi = gen_phi.swapaxes(0, 1)
    gen_pi_att = gen_pi_att.swapaxes(0, 1)

    for i in range(args.num_samples*LATENT_NUM):
        this_num_steps = int(features_mask_tr.sum(axis=0)[i])
        this_labels_length = int(labels_mask_tr.sum(axis=1)[i])
        this_x = gen_x[i][:this_num_steps]
        this_k = gen_k[i][:this_num_steps]
        this_w = gen_w[i][:this_num_steps]
        this_pi = gen_pi[i][:this_num_steps]
        this_phi = gen_phi[i][:this_num_steps, :this_labels_length]
        this_pi_att = gen_pi_att[i][:this_num_steps]

        full_plot(
            [this_x, this_pi_att, this_k, this_w, this_phi],
            os.path.join(
                args.save_dir, 'samples',
                args.samples_name + '_' + str(i/LATENT_NUM) + '_latent_' + str(i%LATENT_NUM) + '.png'))



for i, this_sample in enumerate(gen_x):
    this_sample = this_sample[:int(features_mask_tr.sum(axis=0)[i])]
    generate_wav(
        this_sample,
        os.path.join(args.save_dir, 'samples'),
        args.samples_name + '_' + str(i/LATENT_NUM) + '_latent_' + str(i%LATENT_NUM),
        sptk_dir=args.sptk_dir,
        world_dir=args.world_dir,
        norm_info_file=norm_info_file,
        do_post_filtering=args.do_post_filtering)

if args.process_originals:
    assert not args.new_sentences
    for i, this_sample in enumerate(features_tr.swapaxes(0, 1)):
        this_sample = this_sample[:int(features_mask_tr.sum(axis=0)[i])]
        generate_wav(
            this_sample,
            os.path.join(args.save_dir, 'samples'),
            'original_' + args.samples_name + '_' + str(i/LATENT_NUM) + '_latent_' + str(i%LATENT_NUM),
            sptk_dir=args.sptk_dir,
            world_dir=args.world_dir,
            norm_info_file=norm_info_file,
            do_post_filtering=args.do_post_filtering)


if args.animation:
    if saved_args.labels_type in ['unaligned_phonemes', 'text']:
        for i in range(args.num_samples*LATENT_NUM):
            this_num_steps = int(features_mask_tr.sum(axis=0)[i])
            this_labels_length = int(labels_mask_tr.sum(axis=1)[i])
            this_x = gen_x[i][:this_num_steps]
            this_k = gen_k[i][:this_num_steps]
            this_w = gen_w[i][:this_num_steps]
            this_pi = gen_pi[i][:this_num_steps]
            this_phi = gen_phi[i][:this_num_steps, :this_labels_length]
            this_pi_att = gen_pi_att[i][:this_num_steps]
            create_animation(
                [this_x, this_pi_att, this_k, this_w, this_phi],
                args.samples_name + '_' + str(i/LATENT_NUM) + '_latent_' + str(i%LATENT_NUM) + '.wav',
                args.samples_name + '_' + str(i/LATENT_NUM) + '_latent_' + str(i%LATENT_NUM),
                os.path.join(args.save_dir, 'samples'))

    elif saved_args.labels_type in ['phonemes']:
        for i in range(args.num_samples*LATENT_NUM):
            this_num_steps = int(features_mask_tr.sum(axis=0)[i])
            this_x = gen_x[i][:this_num_steps]
            this_phoneme = labels_tr[:, i][:this_num_steps]
            create_animation(
                [this_x, numpy_one_hot(
                    this_phoneme, saved_args.num_characters)],
                args.samples_name + '_' + str(i/LATENT_NUM) + '_latent_' + str(i%LATENT_NUM) + '.wav',
                args.samples_name + '_' + str(i/LATENT_NUM) + '_latent_' + str(i%LATENT_NUM),
                os.path.join(args.save_dir, 'samples'))

        if args.process_originals:
            for i in range(args.num_samples*LATENT_NUM):
                this_num_steps = int(features_mask_tr.sum(axis=0)[i])
                this_x = features_tr[:, i][:this_num_steps]
                this_phoneme = labels_tr[:, i][:this_num_steps]
                create_animation(
                    [this_x, numpy_one_hot(
                        this_phoneme, saved_args.num_characters)],
                    'original_' + args.samples_name + '_' + str(i/LATENT_NUM) + '_latent_' + str(i%LATENT_NUM) + '.wav',
                    'original_' + '_' + str(i/LATENT_NUM) + '_latent_' + str(i%LATENT_NUM),
                    os.path.join(args.save_dir, 'samples'))
