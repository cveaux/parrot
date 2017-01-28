from blocks.bricks import (
    Initializable, Linear, Random)
from blocks.bricks.base import lazy, application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional
from blocks.roles import add_role, INITIAL_STATE
from blocks.utils import shared_floatx_zeros, dict_union, shared_floatx

import numpy

import theano
from theano import tensor, function

floatX = theano.config.floatX


def _simple_norm(x, eps=1e-5):
    output = (x - tensor.shape_padright(x.mean(-1))) / \
        (eps + tensor.shape_padright(x.std(-1)))
    return output


def _apply_norm(x, layer_norm=True):
    if layer_norm:
        return _simple_norm(x)
    else:
        return x


def logsumexp(x, axis=None):
    x_max = tensor.max(x, axis=axis, keepdims=True)
    z = tensor.log(
        tensor.sum(tensor.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis)


def predict(probs, axis=-1):
    return tensor.argmax(probs, axis=axis)


def kl_unit_gaussian(mu, log_sigma):
    """
    KL divergence from a unit Gaussian prior
    mean across axis 0 (minibatch), sum across all other axes
    based on yaost, via Alec via Ishaan
    """
    return -0.5 * (1 + 2 * log_sigma - mu**2 - tensor.exp(2 * log_sigma))



# https://gist.github.com/benanne/2300591
def one_hot(t, r=None):
    """Compute one hot encoding.

    given a tensor t of dimension d with integer values from range(r), return a
    new tensor of dimension d + 1 with values 0/1, where the last dimension
    gives a one-hot representation of the values in t.
    if r is not given, r is set to max(t) + 1

    """
    if r is None:
        r = tensor.max(t) + 1

    ranges = tensor.shape_padleft(tensor.arange(r), t.ndim)
    return tensor.eq(ranges, tensor.shape_padright(t, 1))


def cost_gmm(y, mu, sig, weight):
    """Gaussian mixture model negative log-likelihood.

    Computes the cost.

    """
    n_dim = y.ndim
    shape_y = y.shape

    k = weight.shape[-1]

    y = y.reshape((-1, shape_y[-1]))
    y = tensor.shape_padright(y)

    mu = mu.reshape((-1, shape_y[-1], k))
    sig = sig.reshape((-1, shape_y[-1], k))
    weight = weight.reshape((-1, k))

    diff = tensor.sqr(y - mu)

    inner = -0.5 * tensor.sum(
        diff / sig**2 +
        2 * tensor.log(sig) + tensor.log(2 * numpy.pi), axis=-2)

    nll = -logsumexp(tensor.log(weight) + inner, axis=-1)

    return nll.reshape(shape_y[:-1], ndim=n_dim - 1)


def sample_gmm(mu, sigma, weight, theano_rng):

    k = weight.shape[-1]
    dim = mu.shape[-1] / k

    shape_result = weight.shape
    shape_result = tensor.set_subtensor(shape_result[-1], dim)
    ndim_result = weight.ndim

    mu = mu.reshape((-1, dim, k))
    sigma = sigma.reshape((-1, dim, k))
    weight = weight.reshape((-1, k))

    sample_weight = theano_rng.multinomial(pvals=weight, dtype=weight.dtype)
    idx = predict(sample_weight, axis=-1)

    mu = mu[tensor.arange(mu.shape[0]), :, idx]
    sigma = sigma[tensor.arange(sigma.shape[0]), :, idx]

    epsilon = theano_rng.normal(
        size=mu.shape, avg=0., std=1., dtype=mu.dtype)

    result = mu + sigma * epsilon

    return result.reshape(shape_result, ndim=ndim_result)

def sample_softmax(predicted, levels):
    predicted_ = predicted.reshape(predicted.shape[:2] + [predicted.shape[2]//levels, levels])
    predicted_reshaped = predicted_.reshape((-1, levels))
    predicted_levels = tensor.argmax(tensor.nnet.softmax(predicted_reshaped), axis=1)
    output = predicted_levels.reshape(predicted_.shape[:3])
    return output


def compute_cce(predicted, ground_truth, levels):
    predicted_ = predicted.reshape(predicted.shape[:2] + [predicted.shape[2]//levels, levels])
    predicted_reshaped = predicted_.reshape((-1, levels))

    predicted_pvals = tensor.nnet.softmax(predicted_reshaped)
    cost = tensor.categorical_cross_entropy(predicted_pvals, ground_truth.flatten())
    cost = cost.reshape(predicted_.shape[:3])
    return cost


class RecurrentWithFork(Initializable):
    # Obtained from Dima's code. @rizar
    # https://github.com/rizar/attention-lvcsr/blob/master/lvsr/bricks/__init__.py
    @lazy(allocation=['input_dim'])
    def __init__(self, recurrent, input_dim, **kwargs):
        super(RecurrentWithFork, self).__init__(**kwargs)
        self.recurrent = recurrent
        self.input_dim = input_dim
        self.fork = Fork(
            [name for name in self.recurrent.sequences
             if name != 'mask'], prototype=Linear())
        self.children = [recurrent.brick, self.fork]

    def _push_allocation_config(self):
        self.fork.input_dim = self.input_dim
        self.fork.output_dims = [self.recurrent.brick.get_dim(name)
                                 for name in self.fork.output_names]

    @application(inputs=['input_', 'mask'])
    def apply(self, input_, mask=None, **kwargs):
        return self.recurrent(
            mask=mask, **dict_union(self.fork.apply(input_, as_dict=True),
                                    kwargs))

    @apply.property('outputs')
    def apply_outputs(self):
        return self.recurrent.states


class Encoder(Initializable):
    def __init__(
            self,
            encoder_type,
            num_characters,
            input_dim,
            encoder_dim,
            **kwargs):
        assert encoder_type in [None, 'bidirectional']
        self.encoder_type = encoder_type
        super(Encoder, self).__init__(**kwargs)

        self.children = []

        if encoder_type in ['lookup', 'bidirectional']:
            self.embed_label = LookupTable(
                num_characters,
                input_dim,
                name='embed_label')
            self.children += [
                self.embed_label]
        else:
            # If there is no encoder.
            assert num_characters == input_dim

        if encoder_type == 'bidirectional':
            transition = RecurrentWithFork(
                GatedRecurrent(dim=encoder_dim).apply,
                input_dim, name='encoder_transition')
            self.encoder = Bidirectional(transition, name='encoder')
            self.children.append(self.encoder)

    @application
    def apply(self, x, x_mask=None):
        if self.encoder_type is None:
            return x

        if self.encoder_type in ['lookup', 'bidirectional']:
            embed_x = self.embed_label.apply(x)

        if self.encoder_type == 'lookup':
            encoded_x = embed_x

        if self.encoder_type == 'bidirectional':
            encoded_x = self.encoder.apply(embed_x, x_mask)

        return encoded_x

class LatentEncoder(Initializable):
    def __init__(
            self,
            encoder_type,
            input_dim,
            rnn_h_dim,
            latent_dim,
            num_layers,
            **kwargs):

        assert encoder_type in ['unidirectional']

        self.encoder_type = encoder_type
        self.num_layers = num_layers

        self.children = []

        self.rnns = []
        self.linear_transforms = []

        for i in range(num_layers):
            rnn_input_gates = Fork(
                                    output_names=[
                                            'latent_rnn_{}_inputs'.format(i+1),
                                            'latent_rnn_{}_gates'.format(i+1)
                                            ],
                                    input_dim=rnn_h_dim if i != 0 else input_dim,
                                    output_dims=[rnn_h_dim, 2 * rnn_h_dim],
                                    name='latent_rnn_{}_to_{}'.format(i, i+1)
                                )
            rnn = GatedRecurrent(dim=rnn_h_dim, name='latent_rnn_{}'.format(i+1))
            self.children.append(rnn_input_gates)
            self.children.append(rnn)

            self.rnns.append(rnn)
            self.linear_transforms.append(rnn_input_gates)

        mu_sig_fork = Fork(
                            output_names=[
                                    'latent_mu',
                                    'latent_log_sig'
                                    ],
                            input_dim=rnn_h_dim,
                            output_dims=[latent_dim, latent_dim],
                            name='rnn_{}_to_mu_logsig'.format(num_layers)
                        )

        self.children.append(mu_sig_fork)

        self.linear_transforms.append(mu_sig_fork)

        kwargs.setdefault('children', []).extend(self.children)

        super(LatentEncoder, self).__init__(**kwargs)

    def initial_states(self, batch_size):
        
        return [rnn.initial_states(batch_size) for rnn in self.rnns]


    @application(inputs=['x', 'x_mask'], outputs=["latent_mu", "latent_log_sigma"])
    def apply(self, x, x_mask):
        """
        Confirm that the input has shape (timesteps, batch_size, dim)
        """
        next_x = x
        for i in range(self.num_layers):
            inp, gates = self.linear_transforms[i].apply(next_x)
            next_x = self.rnns[i].apply(inp, gates)

        next_x = next_x*x_mask[:,:,None]

        raw_stats = next_x.sum(axis=0)/(x_mask.sum(axis=0)[:, None] + 1e-5)

        latent_mu, latent_log_sigma = self.linear_transforms[self.num_layers].apply(raw_stats)

        return latent_mu, latent_log_sigma


class Parrot(Initializable, Random):
    def __init__(
            self,
            input_dim=420,  # Dimension of the text labels
            output_dim=63,  # Dimension of vocoder fram
            rnn_h_dim=1024,  # Size of rnn hidden state
            readouts_dim=1024,  # Size of readouts (summary of rnn)
            labels_type='full_labels',  # full or phoneme labels
            weak_feedback=False,  # Feedback to the top rnn layer
            full_feedback=False,  # Feedback to all rnn layers
            very_weak_feedback=False,  # Feedback to only h3
            feedback_noise_level=None,  # Amount of noise in feedback
            layer_norm=False,  # Use simple normalization?
            use_speaker=False,  # Condition on the speaker id?
            num_speakers=21,  # How many speakers there are?
            speaker_dim=128,  # Size of speaker embedding
            which_cost='MSE',  # Train with MSE or GMM or CCE (categorical_cross_entropy)
            k_gmm=20,  # How many components in the GMM
            levels=256,
            sampling_bias=0,  # Make samples more likely (Graves13)
            epsilon=1e-5,  # Numerical stabilities
            num_characters=43,  # how many chars in the labels
            attention_type='graves',  # graves or softmax
            attention_size=10,  # number of gaussians in the attention
            attention_alignment=1.,  # audio steps per letter at initialization
            sharpening_coeff=1.,
            timing_coeff=1.,
            encoder_type=None,
            encoder_dim=128,
            output_embed_dim=8,
            use_latent=True,
            latent_dim=64,
            initial_iters=0,
            use_mutual_info=False,
            only_noise=False,
            only_residual_train=False,
            only_compute_delta_norm=True,
            quantized_input=False,
            **kwargs):

        super(Parrot, self).__init__(**kwargs)

        if labels_type in ['unaligned_phonemes', 'text']:
            labels_type = 'unaligned'

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_h_dim = rnn_h_dim
        self.readouts_dim = readouts_dim
        self.labels_type = labels_type
        self.layer_norm = layer_norm
        self.which_cost = which_cost
        self.use_speaker = use_speaker
        self.full_feedback = full_feedback
        self.feedback_noise_level = feedback_noise_level
        self.epsilon = epsilon

        self.num_characters = num_characters
        self.attention_type = attention_type
        self.attention_alignment = attention_alignment
        self.attention_size = attention_size
        self.sharpening_coeff = sharpening_coeff
        self.timing_coeff = timing_coeff

        self.encoder_type = encoder_type
        self.encoder_dim = encoder_dim

        self.encoded_input_dim = input_dim
        self.latent_dim = latent_dim
        self.initial_iters = initial_iters

        self.use_mutual_info = use_mutual_info
        self.very_weak_feedback = very_weak_feedback
        self.only_noise = only_noise
        self.only_residual_train = only_residual_train

        self.only_compute_delta_norm = False

        self.quantized_input = quantized_input
        self.output_embed_dim = output_embed_dim

        if self.very_weak_feedback:
            self.weak_feedback = False
            self.full_feedback = False


        """TODO: Verify wherever use_latent has been used"""
        self.use_latent = use_latent  

        if self.encoder_type == 'bidirectional':
            self.encoded_input_dim = 2 * encoder_dim

        assert labels_type in [
            'full_labels', 'phonemes', 'unconditional', 'unaligned']

        if self.feedback_noise_level is not None:
            self.noise_level_var = tensor.scalar('feedback_noise_level')

        if self.quantized_input:
            self.output_embed = LookupTable(
                    levels,
                    self.output_embed_dim,
                    name='output_embed')
            self.embed_to_usual = Linear(
                input_dim=levels*self.output_embed_dim,
                output_dim=self.output_dim,
                name="embed_to_usual")

        self.rnn1 = GatedRecurrent(dim=rnn_h_dim, name='rnn1')
        self.rnn2 = GatedRecurrent(dim=rnn_h_dim, name='rnn2')
        self.rnn3 = GatedRecurrent(dim=rnn_h_dim, name='rnn3')

        self.h1_to_readout = Linear(
            input_dim=rnn_h_dim,
            output_dim=readouts_dim,
            name='h1_to_readout')

        self.h2_to_readout = Linear(
            input_dim=rnn_h_dim,
            output_dim=readouts_dim,
            name='h2_to_readout')

        self.h3_to_readout = Linear(
            input_dim=rnn_h_dim,
            output_dim=readouts_dim,
            name='h3_to_readout')

        self.h1_to_h2 = Fork(
            output_names=['rnn2_inputs', 'rnn2_gates'],
            input_dim=rnn_h_dim,
            output_dims=[rnn_h_dim, 2 * rnn_h_dim],
            name='h1_to_h2')

        self.h1_to_h3 = Fork(
            output_names=['rnn3_inputs', 'rnn3_gates'],
            input_dim=rnn_h_dim,
            output_dims=[rnn_h_dim, 2 * rnn_h_dim],
            name='h1_to_h3')

        self.h2_to_h3 = Fork(
            output_names=['rnn3_inputs', 'rnn3_gates'],
            input_dim=rnn_h_dim,
            output_dims=[rnn_h_dim, 2 * rnn_h_dim],
            name='h2_to_h3')


        if which_cost == 'MSE':
            self.readout_to_output = Linear(
                input_dim=readouts_dim,
                output_dim=output_dim,
                name='readout_to_output')
        elif which_cost == 'GMM':
            self.sampling_bias = sampling_bias
            self.k_gmm = k_gmm
            self.readout_to_output = Fork(
                output_names=['gmm_mu', 'gmm_sigma', 'gmm_coeff'],
                input_dim=readouts_dim,
                output_dims=[output_dim * k_gmm, output_dim * k_gmm, k_gmm],
                name='readout_to_output')
        elif which_cost == 'CCE':
            self.levels = levels
            self.readout_to_output = Linear(
                input_dim=readouts_dim,
                output_dim=output_dim,
                name='readout_to_output')

        self.encoder = Encoder(
            encoder_type,
            num_characters,
            input_dim,
            encoder_dim,
            name='encoder')

        self.children = [
            self.encoder,
            self.rnn1,
            self.rnn2,
            self.rnn3,
            self.h1_to_readout,
            self.h2_to_readout,
            self.h3_to_readout,
            self.h1_to_h2,
            self.h1_to_h3,
            self.h2_to_h3,
            self.readout_to_output]

        if self.use_latent:
            self.latent_encoder = LatentEncoder(
                "unidirectional",
                output_dim,
                rnn_h_dim,
                latent_dim,
                3,
                **kwargs)
            self.children += [
                self.latent_encoder]

        if labels_type != 'unconditional':
            self.inp_to_h1 = Fork(
                output_names=['rnn1_inputs', 'rnn1_gates'],
                input_dim=self.encoded_input_dim,
                output_dims=[rnn_h_dim, 2 * rnn_h_dim],
                name='inp_to_h1')

            self.inp_to_h2 = Fork(
                output_names=['rnn2_inputs', 'rnn2_gates'],
                input_dim=self.encoded_input_dim,
                output_dims=[rnn_h_dim, 2 * rnn_h_dim],
                name='inp_to_h2')

            self.inp_to_h3 = Fork(
                output_names=['rnn3_inputs', 'rnn3_gates'],
                input_dim=self.encoded_input_dim,
                output_dims=[rnn_h_dim, 2 * rnn_h_dim],
                name='inp_to_h3')

            self.children += [
                self.inp_to_h1,
                self.inp_to_h2,
                self.inp_to_h3]

            if labels_type == 'phonemes':
                self.embed_label = LookupTable(
                    num_characters,
                    self.encoded_input_dim,
                    name='embed_label')
                self.children += [
                    self.embed_label]

            if labels_type == 'unaligned':
                # assert num_characters == input_dim

                self.h1_to_att = Fork(
                    output_names=['alpha', 'beta', 'kappa'],
                    input_dim=rnn_h_dim,
                    output_dims=[attention_size] * 3,
                    name='h1_to_att')

                self.att_to_readout = Linear(
                    input_dim=self.encoded_input_dim,
                    output_dim=readouts_dim,
                    name='att_to_readout')

                self.children += [
                    self.h1_to_att,
                    self.att_to_readout]

        if use_speaker:
            self.num_speakers = num_speakers
            self.speaker_dim = speaker_dim
            self.embed_speaker = LookupTable(num_speakers, speaker_dim)

            self.speaker_to_h1 = Fork(
                output_names=['rnn1_inputs', 'rnn1_gates'],
                input_dim=speaker_dim,
                output_dims=[rnn_h_dim, 2 * rnn_h_dim],
                name='speaker_to_h1')

            self.speaker_to_h2 = Fork(
                output_names=['rnn2_inputs', 'rnn2_gates'],
                input_dim=speaker_dim,
                output_dims=[rnn_h_dim, 2 * rnn_h_dim],
                name='speaker_to_h2')

            self.speaker_to_h3 = Fork(
                output_names=['rnn3_inputs', 'rnn3_gates'],
                input_dim=speaker_dim,
                output_dims=[rnn_h_dim, 2 * rnn_h_dim],
                name='speaker_to_h3')

            self.speaker_to_readout = Linear(
                input_dim=speaker_dim,
                output_dim=readouts_dim,
                name='speaker_to_readout')

            if which_cost == 'MSE':
                self.speaker_to_output = Linear(
                    input_dim=speaker_dim,
                    output_dim=output_dim,
                    name='speaker_to_output')
            elif which_cost == 'GMM':
                self.speaker_to_output = Fork(
                    output_names=['gmm_mu', 'gmm_sigma', 'gmm_coeff'],
                    input_dim=speaker_dim,
                    output_dims=[
                        output_dim * k_gmm, output_dim * k_gmm, k_gmm],
                    name='speaker_to_output')
            elif which_cost == 'CCE':
                self.speaker_to_output = Linear(
                    input_dim=speaker_dim,
                    output_dim=output_dim*levels,
                    name='speaker_to_output')

            self.children += [
                self.embed_speaker,
                self.speaker_to_h1,
                self.speaker_to_h2,
                self.speaker_to_h3,
                self.speaker_to_readout,
                self.speaker_to_output]

        if self.use_latent:
            self.latent_to_h = Fork(
                output_names=['rnn1_inputs', 'rnn1_gates', 'rnn2_inputs',
                             'rnn2_gates', 'rnn3_inputs', 'rnn3_gates' ],
                input_dim=latent_dim,
                output_dims=[rnn_h_dim, 2 * rnn_h_dim, rnn_h_dim, 
                                2 * rnn_h_dim, rnn_h_dim, 2 * rnn_h_dim],
                name='latent_to_h')

            self.latent_to_readout = Linear(
                input_dim=latent_dim,
                output_dim=readouts_dim,
                name='latent_to_readout')

            if which_cost == 'MSE':
                self.latent_to_output = Linear(
                    input_dim=latent_dim,
                    output_dim=output_dim,
                    name='speaker_to_output')
            elif which_cost == 'GMM':
                self.latent_to_output = Fork(
                    output_names=['gmm_mu', 'gmm_sigma', 'gmm_coeff'],
                    input_dim=latent_dim,
                    output_dims=[
                        output_dim * k_gmm, output_dim * k_gmm, k_gmm],
                    name='speaker_to_output')
            elif which_cost == 'CCE':
                self.latent_to_output = Linear(
                    input_dim=latent_dim,
                    output_dim=output_dim*levels,
                    name='speaker_to_output')

            self.children += [
                self.latent_to_h,
                self.latent_to_readout,
                self.latent_to_output]


        if full_feedback:
            self.out_to_h2 = Fork(
                output_names=['rnn2_inputs', 'rnn2_gates'],
                input_dim=output_dim,
                output_dims=[rnn_h_dim, 2 * rnn_h_dim],
                name='out_to_h2')

            self.out_to_h3 = Fork(
                output_names=['rnn3_inputs', 'rnn3_gates'],
                input_dim=output_dim,
                output_dims=[rnn_h_dim, 2 * rnn_h_dim],
                name='out_to_h3')
            self.children += [
                self.out_to_h2,
                self.out_to_h3]
            weak_feedback = True

        self.weak_feedback = weak_feedback

        if weak_feedback:
            self.out_to_h1 = Fork(
                output_names=['rnn1_inputs', 'rnn1_gates'],
                input_dim=output_dim,
                output_dims=[rnn_h_dim, 2 * rnn_h_dim],
                name='out_to_h1')
            self.children += [
                self.out_to_h1]

        if self.very_weak_feedback:
            self.out_to_h3_feedback = Fork(
                output_names=['rnn3_inputs_feedback', 'rnn3_gates_feedback'],
                input_dim=output_dim,
                output_dims=[rnn_h_dim, 2 * rnn_h_dim],
                name='out_to_h3_feedback')
            self.children += [
                self.out_to_h3_feedback]

    def _allocate(self):
        self.initial_w = shared_floatx_zeros(
            (self.encoded_input_dim,), name="initial_w")

        add_role(self.initial_w, INITIAL_STATE)

    def symbolic_input_variables(self):
        if self.quantized_input:
            features = tensor.itensor3('features')
        else:
            features = tensor.tensor3('features')
        features_mask = tensor.matrix('features_mask')

        if self.labels_type == 'full_labels':
            labels = tensor.tensor3('labels')
        elif self.labels_type in ['phonemes', 'unaligned']:
            labels = tensor.imatrix('labels')
        elif self.labels_type == 'unconditional':
            labels = None

        if self.labels_type == 'unaligned':
            labels_mask = tensor.matrix('labels_mask')
        else:
            # Maybe I should define labels_mask as features_mask
            # labels_mask = None
            labels_mask = features_mask

        start_flag = tensor.scalar('start_flag')

        if self.use_speaker:
            speaker = tensor.imatrix('speaker_index')
        else:
            speaker = None

        if self.use_latent:
            latent_var = tensor.matrix('latent_code')
        else:
            latent_var = None

        return features, features_mask, labels, labels_mask, \
            speaker, latent_var, start_flag

    def initial_states(self, batch_size):
        initial_h1 = self.rnn1.initial_states(batch_size)
        initial_h2 = self.rnn2.initial_states(batch_size)
        initial_h3 = self.rnn3.initial_states(batch_size)

        last_h1 = shared_floatx_zeros((batch_size, self.rnn_h_dim))
        last_h2 = shared_floatx_zeros((batch_size, self.rnn_h_dim))
        last_h3 = shared_floatx_zeros((batch_size, self.rnn_h_dim))

        # Defining for all
        initial_k = tensor.zeros(
            (batch_size, self.attention_size), dtype=floatX)
        last_k = shared_floatx_zeros((batch_size, self.attention_size))

        # Trainabla initial state for w. Why not for k?
        if self.labels_type != 'unaligned':
            initial_w = tensor.zeros(
                (batch_size, self.encoded_input_dim), dtype=floatX)
        else:
            initial_w = tensor.repeat(self.initial_w[None, :], batch_size, 0)

        last_w = shared_floatx_zeros((batch_size, self.encoded_input_dim))

        return initial_h1, last_h1, initial_h2, last_h2, initial_h3, last_h3, \
            initial_w, last_w, initial_k, last_k

    @application
    def compute_cost(
            self, features, features_mask, labels, labels_mask,
            speaker, start_flag, batch_size, is_train=True):

        if speaker is None:
            assert not self.use_speaker

        if labels is None:
            assert self.labels_type == 'unconditional'

        kl_cost = None
        mutual_info = None

        target_features = features[1:]
        mask = features_mask[1:]

        if self.quantized_input:
            features = self.ouput_embed.apply(features)
            features = features.reshape(features.shape[:2] + [-1])
            features = self.embed_to_usual.apply(features)

        cell_shape = (mask.shape[0], batch_size, self.rnn_h_dim)
        gat_shape = (mask.shape[0], batch_size, 2 * self.rnn_h_dim)
        cell_h1 = tensor.zeros(cell_shape, dtype=floatX)
        cell_h2 = tensor.zeros(cell_shape, dtype=floatX)
        cell_h3 = tensor.zeros(cell_shape, dtype=floatX)
        gat_h1 = tensor.zeros(gat_shape, dtype=floatX)
        gat_h2 = tensor.zeros(gat_shape, dtype=floatX)
        gat_h3 = tensor.zeros(gat_shape, dtype=floatX)

        if self.labels_type not in ['unconditional', 'unaligned']:
            labels = labels[1:]

            if self.labels_type == 'phonemes':
                labels = self.embed_label.apply(labels)

            inp_cell_h1, inp_gat_h1 = self.inp_to_h1.apply(labels)
            inp_cell_h2, inp_gat_h2 = self.inp_to_h2.apply(labels)
            inp_cell_h3, inp_gat_h3 = self.inp_to_h3.apply(labels)

            to_normalize = [
                inp_cell_h1, inp_gat_h1, inp_cell_h2, inp_gat_h2,
                inp_cell_h3, inp_gat_h3]

            inp_cell_h1, inp_gat_h1, inp_cell_h2, inp_gat_h2, \
                inp_cell_h3, inp_gat_h3 = \
                [_apply_norm(x, self.layer_norm) for x in to_normalize]

            cell_h1 += inp_cell_h1
            cell_h2 += inp_cell_h2
            cell_h3 += inp_cell_h3
            gat_h1 += inp_gat_h1
            gat_h2 += inp_gat_h2
            gat_h3 += inp_gat_h3

        input_features = features[:-1]
        if self.weak_feedback:
            if self.only_noise:
                input_features = tensor.zeros_like(input_features)

            if self.feedback_noise_level:
                noise = self.theano_rng.normal(
                    size=input_features.shape,
                    avg=0., std=1.)
                
                input_features += self.noise_level_var * noise

            out_cell_h1, out_gat_h1 = self.out_to_h1.apply(input_features)

            to_normalize = [
                out_cell_h1, out_gat_h1]
            out_cell_h1, out_gat_h1 = \
                [_apply_norm(x, self.layer_norm) for x in to_normalize]

            cell_h1 += out_cell_h1
            gat_h1 += out_gat_h1

        if self.full_feedback:
            assert self.weak_feedback
            out_cell_h2, out_gat_h2 = self.out_to_h2.apply(input_features)
            out_cell_h3, out_gat_h3 = self.out_to_h3.apply(input_features)

            to_normalize = [
                out_cell_h2, out_gat_h2, out_cell_h3, out_gat_h3]
            out_cell_h2, out_gat_h2, out_cell_h3, out_gat_h3 = \
                [_apply_norm(x, self.layer_norm) for x in to_normalize]

            cell_h2 += out_cell_h2
            gat_h2 += out_gat_h2
            cell_h3 += out_cell_h3
            gat_h3 += out_gat_h3

        if self.very_weak_feedback:
            input_features = features[:-1]

            if self.only_noise:
                input_features = tensor.zeros_like(input_features)


            if self.feedback_noise_level:
                noise = self.theano_rng.normal(
                    size=input_features.shape,
                    avg=0., std=1.)

                input_features += self.noise_level_var * noise

            out_cell_h3, out_gat_h3 = self.out_to_h3_feedback.apply(input_features)
            to_normalize = [out_cell_h3, out_gat_h3]
            out_cell_h3, out_gat_h3 = \
                [_apply_norm(x, self.layer_norm) for x in to_normalize]
            cell_h3 += out_cell_h3
            gat_h3 += out_gat_h3

        if self.use_speaker:
            speaker = speaker[:, 0]
            emb_speaker = self.embed_speaker.apply(speaker)
            emb_speaker = tensor.shape_padleft(emb_speaker)

            spk_cell_h1, spk_gat_h1 = self.speaker_to_h1.apply(emb_speaker)
            spk_cell_h2, spk_gat_h2 = self.speaker_to_h2.apply(emb_speaker)
            spk_cell_h3, spk_gat_h3 = self.speaker_to_h3.apply(emb_speaker)

            to_normalize = [
                spk_cell_h1, spk_gat_h1, spk_cell_h2, spk_gat_h2,
                spk_cell_h3, spk_gat_h3]

            spk_cell_h1, spk_gat_h1, spk_cell_h2, spk_gat_h2, \
                spk_cell_h3, spk_gat_h3, = \
                [_apply_norm(x, self.layer_norm) for x in to_normalize]

            cell_h1 = spk_cell_h1 + cell_h1
            cell_h2 = spk_cell_h2 + cell_h2
            cell_h3 = spk_cell_h3 + cell_h3
            gat_h1 = spk_gat_h1 + gat_h1
            gat_h2 = spk_gat_h2 + gat_h2
            gat_h3 = spk_gat_h3 + gat_h3

        if self.use_latent:
            mu, log_sig = self.latent_encoder.apply(features, features_mask)
            e = tensor.cast(self.theano_rng.normal(mu.shape), floatX)

            latent_var = mu + e*(tensor.exp(log_sig) + self.epsilon)

            kl_cost = kl_unit_gaussian(mu, log_sig).sum(axis=1)

            kl_cost = kl_cost.mean()

            latent_var = tensor.shape_padleft(latent_var)


            latent_cell_h1, latent_gat_h1, latent_cell_h2, \
                latent_gat_h2, latent_cell_h3, latent_gat_h3 = self.latent_to_h.apply(latent_var)

            # to_normalize = [
            #     latent_cell_h1, latent_gat_h1, latent_cell_h2,
            #     latent_gat_h2, latent_cell_h3, latent_gat_h3 ]

            # latent_cell_h1, latent_gat_h1, latent_cell_h2, \
            #     latent_gat_h2, latent_cell_h3, latent_gat_h3 = \
            #     [_apply_norm(x, self.layer_norm) for x in to_normalize]

            cell_h1 = latent_cell_h1 + cell_h1
            cell_h2 = latent_cell_h2 + cell_h2
            cell_h3 = latent_cell_h3 + cell_h3
            gat_h1 = latent_gat_h1 + gat_h1
            gat_h2 = latent_gat_h2 + gat_h2
            gat_h3 = latent_gat_h3 + gat_h3



        initial_h1, last_h1, initial_h2, last_h2, initial_h3, last_h3, \
            initial_w, last_w, initial_k, last_k = \
            self.initial_states(batch_size)

        # If it's a new example, use initial states.
        input_h1 = tensor.switch(
            start_flag, initial_h1, last_h1)
        input_h2 = tensor.switch(
            start_flag, initial_h2, last_h2)
        input_h3 = tensor.switch(
            start_flag, initial_h3, last_h3)
        input_w = tensor.switch(
            start_flag, initial_w, last_w)
        input_k = tensor.switch(
            start_flag, initial_k, last_k)

        if self.labels_type == 'unaligned':
            # TODO: include context_oh as context in the step function.
            if self.encoder_type is None:
                context_oh = one_hot(labels, self.num_characters) * \
                    tensor.shape_padright(labels_mask)
            elif self.encoder_type == 'bidirectional':
                context_oh = self.encoder.apply(labels) * \
                    tensor.shape_padright(labels_mask)

            u = tensor.shape_padleft(
                tensor.arange(labels.shape[1], dtype=floatX), 2)

        def step(
                inp_h1_t, gat_h1_t, inp_h2_t, gat_h2_t, inp_h3_t, gat_h3_t,
                h1_tm1, h2_tm1, h3_tm1, k_tm1, w_tm1, context_oh):

            if self.labels_type == 'unaligned':
                attinp_h1, attgat_h1 = self.inp_to_h1.apply(w_tm1)
                inp_h1_t += attinp_h1
                gat_h1_t += attgat_h1

            h1_t = self.rnn1.apply(
                inp_h1_t,
                gat_h1_t,
                h1_tm1, iterate=False)

            if self.labels_type == 'unaligned':
                a_t, b_t, k_t = self.h1_to_att.apply(h1_t)

                if self.attention_type == "softmax":
                    a_t = tensor.nnet.softmax(a_t) + self.epsilon
                else:
                    a_t = tensor.exp(a_t) + self.epsilon

                b_t = tensor.exp(b_t) + self.epsilon
                k_t = k_tm1 + self.attention_alignment * tensor.exp(k_t)

                a_t_ = a_t
                a_t = tensor.shape_padright(a_t)
                b_t = tensor.shape_padright(b_t)
                k_t_ = tensor.shape_padright(k_t)

                # batch size X att size X len context
                if self.attention_type == "softmax":
                    # numpy.sqrt(1/(2*numpy.pi)) is the weird number
                    phi_t = 0.3989422917366028 * tensor.sum(
                        a_t * tensor.sqrt(b_t) *
                        tensor.exp(-0.5 * b_t * (k_t_ - u)**2), axis=1)
                else:
                    phi_t = tensor.sum(
                        a_t * tensor.exp(-b_t * (k_t_ - u)**2), axis=1)

                # batch size X len context X num letters
                w_t = (tensor.shape_padright(phi_t) * context_oh).sum(axis=1)

                attinp_h2, attgat_h2 = self.inp_to_h2.apply(w_t)
                attinp_h3, attgat_h3 = self.inp_to_h3.apply(w_t)
                inp_h2_t += attinp_h2
                gat_h2_t += attgat_h2
                inp_h3_t += attinp_h3
                gat_h3_t += attgat_h3
            else:
                k_t = k_tm1
                w_t = w_tm1
                a_t_ = k_tm1
                phi_t = k_tm1

            h1inp_h2, h1gat_h2 = self.h1_to_h2.apply(h1_t)
            h1inp_h3, h1gat_h3 = self.h1_to_h3.apply(h1_t)

            to_normalize = [
                h1inp_h2, h1gat_h2, h1inp_h3, h1gat_h3]
            h1inp_h2, h1gat_h2, h1inp_h3, h1gat_h3 = \
                [_apply_norm(x, self.layer_norm) for x in to_normalize]

            h2_t = self.rnn2.apply(
                inp_h2_t + h1inp_h2,
                gat_h2_t + h1gat_h2,
                h2_tm1, iterate=False)

            h2inp_h3, h2gat_h3 = self.h2_to_h3.apply(h2_t)

            to_normalize = [
                h2inp_h3, h2gat_h3]
            h2inp_h3, h2gat_h3 = \
                [_apply_norm(x, self.layer_norm) for x in to_normalize]

            h3_t = self.rnn3.apply(
                inp_h3_t + h1inp_h3 + h2inp_h3,
                gat_h3_t + h1gat_h3 + h2gat_h3,
                h3_tm1, iterate=False)

            return h1_t, h2_t, h3_t, k_t, w_t, phi_t, a_t_

        (h1, h2, h3, k, w, phi, pi_att), scan_updates = theano.scan(
            fn=step,
            sequences=[cell_h1, gat_h1, cell_h2, gat_h2, cell_h3, gat_h3],
            non_sequences=[context_oh],
            outputs_info=[
                input_h1,
                input_h2,
                input_h3,
                input_k,
                input_w,
                None,
                None])

        h1_out = self.h1_to_readout.apply(h1)
        h2_out = self.h2_to_readout.apply(h2)
        h3_out = self.h3_to_readout.apply(h3)

        to_normalize = [
            h1_out, h2_out, h3_out]
        h1_out, h2_out, h3_out = \
            [_apply_norm(x, self.layer_norm) for x in to_normalize]

        readouts = h1_out + h2_out + h3_out

        if self.use_speaker:
            readouts += self.speaker_to_readout.apply(emb_speaker)

        if self.use_latent:
            readouts += self.latent_to_readout.apply(latent_var)

        if self.labels_type == 'unaligned':
            readouts += self.att_to_readout.apply(w)

        predicted = self.readout_to_output.apply(readouts)

        if self.which_cost == 'MSE':
            if self.use_speaker:
                predicted += self.speaker_to_output.apply(emb_speaker)

            if self.use_latent:
                predicted += self.latent_to_output.apply(latent_var)
                if self.use_mutual_info:
                    mu_pred, log_sigma_pred = self.latent_encoder.apply(predicted, mask)
                    mutual_info = 0.5*(2*log_sig - 2*log_sigma_pred + (tensor.exp(2*log_sigma_pred) +\
                                     (mu_pred - mu)**2)*(tensor.exp(-2*log_sig) + 0.00001))

                    mutual_info = mutual_info.sum(axis=1).mean()

            if self.only_compute_delta_norm:
                predicted *= 0.

            if self.only_residual_train:
                predicted += input_features

            cost = tensor.sum((predicted - target_features) ** 2, axis=-1)

            next_x = predicted
            # Dummy value for coeff
            coeff = predicted
        elif self.which_cost == 'GMM':
            mu, sigma, coeff = predicted
            if self.only_residual_train:
                mu += input_features
            if self.use_speaker:
                spk_to_out = self.speaker_to_output.apply(emb_speaker)
                mu += spk_to_out[0]
                sigma += spk_to_out[1]
                coeff += spk_to_out[2]

            if self.use_latent:
                latent_to_out = self.latent_to_output.apply(latent_var)
                mu += latent_to_out[0]
                sigma += latent_to_out[1]
                coeff += latent_to_out[2]

            # When training there should not be sampling_bias
            sigma = tensor.exp(sigma) + self.epsilon

            coeff = tensor.nnet.softmax(
                coeff.reshape(
                    (-1, self.k_gmm))).reshape(
                        coeff.shape) + self.epsilon

            cost = cost_gmm(target_features, mu, sigma, coeff)
            next_x = sample_gmm(mu, sigma, coeff, self.theano_rng)

        elif self.which_cost == 'CCE':
            if self.use_speaker:
                predicted += self.speaker_to_output.apply(emb_speaker)

            if self.use_latent:
                predicted += self.latent_to_output.apply(latent_var)

            cost = compute_cce(predicted, target_features, self.levels)

        cost = (cost * mask).sum() / (mask.sum() + 1e-5) + 0. * start_flag

        if self.use_latent:
            iters = shared_floatx(self.initial_iters)
            # kl_coeff = iters/(iters + 5e5)
            kl_cost = kl_cost/(mask.sum() + 1e-5)
            # cost += kl_coeff*kl_cost
            cost += kl_cost

            if self.use_mutual_info:
                cost += mutual_info/(mask.sum() + 1e-5)


        updates = []
        updates.append((last_h1, h1[-1]))
        updates.append((last_h2, h2[-1]))
        updates.append((last_h3, h3[-1]))

        if self.use_latent:
            updates.append((iters, iters + 1.))

        if self.labels_type == 'unaligned':
            updates.append((last_k, k[-1]))
            updates.append((last_w, w[-1]))

        attention_vars = [next_x, k, w, coeff, phi, pi_att]

        return cost, scan_updates + updates, attention_vars, kl_cost, mutual_info

    @application
    def sample_model_fun(
            self, labels, labels_mask, speaker, latent_var, num_samples, seq_size):

        initial_h1, last_h1, initial_h2, last_h2, initial_h3, last_h3, \
            initial_w, last_w, initial_k, last_k = \
            self.initial_states(num_samples)

        if self.quantized_input:
            initial_x = numpy.zeros(
                (num_samples, self.output_dim), dtype=numpy.int32) + (self.levels - 1)//2
            initial_x = self.input_embed.apply(initial_x)
            initial_x = initial_x.reshape((initial_x.shape[0], -1))
            initial_x = self.embed_to_usual.apply(initial_x)
        else:
            initial_x = numpy.zeros(
                (num_samples, self.output_dim), dtype=floatX)

        cell_shape = (seq_size, num_samples, self.rnn_h_dim)
        gat_shape = (seq_size, num_samples, 2 * self.rnn_h_dim)
        cell_h1 = tensor.zeros(cell_shape, dtype=floatX)
        cell_h2 = tensor.zeros(cell_shape, dtype=floatX)
        cell_h3 = tensor.zeros(cell_shape, dtype=floatX)
        gat_h1 = tensor.zeros(gat_shape, dtype=floatX)
        gat_h2 = tensor.zeros(gat_shape, dtype=floatX)
        gat_h3 = tensor.zeros(gat_shape, dtype=floatX)

        if self.labels_type not in ['unconditional', 'unaligned']:
            if self.labels_type == 'phonemes':
                labels = self.embed_label.apply(labels)

            inp_cell_h1, inp_gat_h1 = self.inp_to_h1.apply(labels)
            inp_cell_h2, inp_gat_h2 = self.inp_to_h2.apply(labels)
            inp_cell_h3, inp_gat_h3 = self.inp_to_h3.apply(labels)

            to_normalize = [
                inp_cell_h1, inp_gat_h1, inp_cell_h2, inp_gat_h2,
                inp_cell_h3, inp_gat_h3]

            inp_cell_h1, inp_gat_h1, inp_cell_h2, inp_gat_h2, \
                inp_cell_h3, inp_gat_h3 = \
                [_apply_norm(x, self.layer_norm) for x in to_normalize]

            cell_h1 += inp_cell_h1
            cell_h2 += inp_cell_h2
            cell_h3 += inp_cell_h3
            gat_h1 += inp_gat_h1
            gat_h2 += inp_gat_h2
            gat_h3 += inp_gat_h3

        if self.use_speaker:
            speaker = speaker[:, 0]
            emb_speaker = self.embed_speaker.apply(speaker)

            # Applied before the broadcast.
            spk_readout = self.speaker_to_readout.apply(emb_speaker)
            spk_output = self.speaker_to_output.apply(emb_speaker)

            # Add dimension to repeat with time.
            emb_speaker = tensor.shape_padleft(emb_speaker)

            spk_cell_h1, spk_gat_h1 = self.speaker_to_h1.apply(emb_speaker)
            spk_cell_h2, spk_gat_h2 = self.speaker_to_h2.apply(emb_speaker)
            spk_cell_h3, spk_gat_h3 = self.speaker_to_h3.apply(emb_speaker)

            to_normalize = [
                spk_cell_h1, spk_gat_h1, spk_cell_h2, spk_gat_h2,
                spk_cell_h3, spk_gat_h3]

            spk_cell_h1, spk_gat_h1, spk_cell_h2, spk_gat_h2, \
                spk_cell_h3, spk_gat_h3, = \
                [_apply_norm(x, self.layer_norm) for x in to_normalize]

            cell_h1 += spk_cell_h1
            cell_h2 += spk_cell_h2
            cell_h3 += spk_cell_h3
            gat_h1 += spk_gat_h1
            gat_h2 += spk_gat_h2
            gat_h3 += spk_gat_h3

        if self.use_latent:
            # latent_var = tensor.cast(self.theano_rng.normal((num_samples, self.latent_dim)), floatX)

            latent_readout = self.latent_to_readout.apply(latent_var)
            latent_output = self.latent_to_output.apply(latent_var)

            latent_var = tensor.shape_padleft(latent_var)

            latent_cell_h1, latent_gat_h1, latent_cell_h2, \
                latent_gat_h2, latent_cell_h3, latent_gat_h3 = self.latent_to_h.apply(latent_var)

            # to_normalize = [
            #     latent_cell_h1, latent_gat_h1, latent_cell_h2,
            #     latent_gat_h2, latent_cell_h3, latent_gat_h3 ]

            # latent_cell_h1, latent_gat_h1, latent_cell_h2, \
            #     latent_gat_h2, latent_cell_h3, latent_gat_h3 = \
            #     [_apply_norm(x, self.layer_norm) for x in to_normalize]

            cell_h1 = latent_cell_h1 + cell_h1
            cell_h2 = latent_cell_h2 + cell_h2
            cell_h3 = latent_cell_h3 + cell_h3
            gat_h1 = latent_gat_h1 + gat_h1
            gat_h2 = latent_gat_h2 + gat_h2
            gat_h3 = latent_gat_h3 + gat_h3



        if self.labels_type == 'unaligned':
            # TODO: include context_oh as context in the step function.
            # batch_size * seq_length * num_characters

            if self.encoder_type is None:
                context_oh = one_hot(labels, self.num_characters) * \
                    tensor.shape_padright(labels_mask)
            elif self.encoder_type == 'bidirectional':
                context_oh = self.encoder.apply(labels) * \
                    tensor.shape_padright(labels_mask)

            u = tensor.shape_padleft(
                tensor.arange(labels.shape[1], dtype=floatX), 2)

        def sample_step(
                inp_cell_h1_t, inp_gat_h1_t, inp_cell_h2_t, inp_gat_h2_t,
                inp_cell_h3_t, inp_gat_h3_t, x_tm1, h1_tm1, h2_tm1, h3_tm1,
                k_tm1, w_tm1):

            cell_h1_t = inp_cell_h1_t
            cell_h2_t = inp_cell_h2_t
            cell_h3_t = inp_cell_h3_t

            gat_h1_t = inp_gat_h1_t
            gat_h2_t = inp_gat_h2_t
            gat_h3_t = inp_gat_h3_t

            if self.only_noise:
                x_tm1 = tensor.zeros_like(x_tm1)

            if self.labels_type == 'unaligned':
                attinp_h1, attgat_h1 = self.inp_to_h1.apply(w_tm1)
                cell_h1_t += attinp_h1
                gat_h1_t += attgat_h1

            if self.weak_feedback:
                out_cell_h1_t, out_gat_h1_t = self.out_to_h1.apply(x_tm1)

                to_normalize = [
                    out_cell_h1_t, out_gat_h1_t]
                out_cell_h1_t, out_gat_h1_t = \
                    [_apply_norm(x, self.layer_norm) for x in to_normalize]

                cell_h1_t += out_cell_h1_t
                gat_h1_t += out_gat_h1_t

            if self.full_feedback:
                out_cell_h2_t, out_gat_h2_t = self.out_to_h2.apply(x_tm1)
                out_cell_h3_t, out_gat_h3_t = self.out_to_h3.apply(x_tm1)

                to_normalize = [
                    out_cell_h2_t, out_gat_h2_t,
                    out_cell_h3_t, out_gat_h3_t]
                out_cell_h2_t, out_gat_h2_t, \
                    out_cell_h3_t, out_gat_h3_t = \
                    [_apply_norm(x, self.layer_norm) for x in to_normalize]

                cell_h2_t += out_cell_h2_t
                cell_h3_t += out_cell_h3_t
                gat_h2_t += out_gat_h2_t
                gat_h3_t += out_gat_h3_t

            if self.very_weak_feedback:
                out_cell_h3_t, out_gat_h3_t = self.out_to_h3_feedback.apply(x_tm1)
                to_normalize = [out_cell_h3_t, out_gat_h3_t]
                out_cell_h3_t, out_gat_h3_t = \
                    [_apply_norm(x, self.layer_norm) for x in to_normalize]
                cell_h3_t += out_cell_h3_t
                gat_h3_t += out_gat_h3_t

            h1_t = self.rnn1.apply(
                cell_h1_t,
                gat_h1_t,
                h1_tm1, iterate=False)

            if self.labels_type == 'unaligned':
                a_t, b_t, k_t = self.h1_to_att.apply(h1_t)

                if self.attention_type == "softmax":
                    a_t = tensor.nnet.softmax(a_t) + self.epsilon
                else:
                    a_t = tensor.exp(a_t) + self.epsilon

                b_t = tensor.exp(b_t) * self.sharpening_coeff + self.epsilon
                k_t = k_tm1 + self.attention_alignment * \
                    tensor.exp(k_t) / self.timing_coeff

                a_t_ = a_t
                a_t = tensor.shape_padright(a_t)
                b_t = tensor.shape_padright(b_t)
                k_t_ = tensor.shape_padright(k_t)

                # batch size X att size X len context
                if self.attention_type == "softmax":
                    # numpy.sqrt(1/(2*numpy.pi)) is the weird number
                    phi_t = 0.3989422917366028 * tensor.sum(
                        a_t * tensor.sqrt(b_t) *
                        tensor.exp(-0.5 * b_t * (k_t_ - u)**2), axis=1)
                else:
                    phi_t = tensor.sum(
                        a_t * tensor.exp(-b_t * (k_t_ - u)**2), axis=1)

                # batch size X len context X num letters
                w_t = (tensor.shape_padright(phi_t) * context_oh).sum(axis=1)

                attinp_h2, attgat_h2 = self.inp_to_h2.apply(w_t)
                attinp_h3, attgat_h3 = self.inp_to_h3.apply(w_t)
                cell_h2_t += attinp_h2
                gat_h2_t += attgat_h2
                cell_h3_t += attinp_h3
                gat_h3_t += attgat_h3
            else:
                k_t = k_tm1
                w_t = w_tm1
                phi_t = k_tm1
                a_t_ = k_tm1

            h1inp_h2, h1gat_h2 = self.h1_to_h2.apply(h1_t)
            h1inp_h3, h1gat_h3 = self.h1_to_h3.apply(h1_t)

            to_normalize = [
                h1inp_h2, h1gat_h2, h1inp_h3, h1gat_h3]
            h1inp_h2, h1gat_h2, h1inp_h3, h1gat_h3 = \
                [_apply_norm(x, self.layer_norm) for x in to_normalize]

            h2_t = self.rnn2.apply(
                cell_h2_t + h1inp_h2,
                gat_h2_t + h1gat_h2,
                h2_tm1, iterate=False)

            h2inp_h3, h2gat_h3 = self.h2_to_h3.apply(h2_t)

            to_normalize = [
                h2inp_h3, h2gat_h3]
            h2inp_h3, h2gat_h3 = \
                [_apply_norm(x, self.layer_norm) for x in to_normalize]

            h3_t = self.rnn3.apply(
                cell_h3_t + h1inp_h3 + h2inp_h3,
                gat_h3_t + h1gat_h3 + h2gat_h3,
                h3_tm1, iterate=False)

            h1_out_t = self.h1_to_readout.apply(h1_t)
            h2_out_t = self.h2_to_readout.apply(h2_t)
            h3_out_t = self.h3_to_readout.apply(h3_t)

            to_normalize = [
                h1_out_t, h2_out_t, h3_out_t]
            h1_out_t, h2_out_t, h3_out_t = \
                [_apply_norm(x, self.layer_norm) for x in to_normalize]

            readout_t = h1_out_t + h2_out_t + h3_out_t

            if self.labels_type == 'unaligned':
                readout_t += self.att_to_readout.apply(w_t)

            if self.use_speaker:
                readout_t += spk_readout

            if self.use_latent:
                readout_t += latent_readout

            output_t = self.readout_to_output.apply(readout_t)

            if self.which_cost == 'MSE':
                predicted_x_t = output_t
                if self.use_speaker:
                    predicted_x_t += spk_output

                if self.use_latent:
                    predicted_x_t += latent_output

                if self.only_compute_delta_norm:
                    predicted_x_t *= 0.
                
                if self.only_residual_train:
                    predicted_x_t += x_tm1


                # Dummy value for coeff_t
                coeff_t = predicted_x_t
            elif self.which_cost == "GMM":
                mu_t, sigma_t, coeff_t = output_t
                if self.use_speaker:
                    mu_t += spk_output[0]
                    sigma_t += spk_output[1]
                    coeff_t += spk_output[2]

                if self.use_latent:
                    mu_t += latent_output[0]
                    sigma_t += latent_output[1]
                    coeff_t += latent_output[2]

                if self.only_residual_train:
                    mu_t += x_tm1

                sigma_t = tensor.exp(sigma_t - self.sampling_bias) + \
                    self.epsilon

                coeff_t = tensor.nnet.softmax(
                    coeff_t.reshape(
                        (-1, self.k_gmm)) * (1. + self.sampling_bias)).reshape(
                            coeff_t.shape) + self.epsilon

                predicted_x_t = sample_gmm(
                    mu_t, sigma_t, coeff_t, self.theano_rng)

            return predicted_x_t, h1_t, h2_t, h3_t, \
                k_t, w_t, coeff_t, phi_t, a_t_

        (sample_x, h1, h2, h3, k, w, pi, phi, pi_att), updates = theano.scan(
            fn=sample_step,
            sequences=[
                cell_h1,
                gat_h1,
                cell_h2,
                gat_h2,
                cell_h3,
                gat_h3],
            non_sequences=[],
            outputs_info=[
                initial_x,
                initial_h1,
                initial_h2,
                initial_h3,
                initial_k,
                initial_w,
                None,
                None,
                None])

        return sample_x, k, w, pi, phi, pi_att, updates

    def sample_model(
            self, labels_tr, labels_mask_tr, features_mask_tr, 
            speaker_tr, latent_var_tr, num_samples):

        features, features_mask, labels, labels_mask, speaker, latent_var, start_flag = \
            self.symbolic_input_variables()

        sample_x, k, w, pi, phi, pi_att, updates = \
            self.sample_model_fun(
                labels, labels_mask, speaker, latent_var,
                num_samples, 1000)

        theano_inputs = []
        numpy_inputs = ()

        if self.labels_type != 'unconditional':
            theano_inputs += [labels]
            numpy_inputs += (labels_tr,)

        if self.labels_type == 'unaligned':
            theano_inputs += [labels_mask]
            numpy_inputs += (labels_mask_tr,)

        if self.use_speaker:
            theano_inputs += [speaker]
            numpy_inputs += (speaker_tr,)

        if self.use_latent:
            theano_inputs += [latent_var]
            numpy_inputs += (latent_var_tr,)

        return function(
            theano_inputs,
            [sample_x, k, w, pi, phi, pi_att],
            updates=updates)(*numpy_inputs)

    def sample_using_input(self, data_tr, num_samples):
        # Used to predict the values using the dataset

        features, features_mask, labels, labels_mask, speaker, latent_var, start_flag = \
            self.symbolic_input_variables()

        cost, updates, attention_vars, kl_cost, mutual_info = self.compute_cost(
            features, features_mask, labels, labels_mask,
            speaker, start_flag, num_samples)
        sample_x, k, w, pi, phi, pi_att = attention_vars

        theano_vars = [
            features, features_mask, labels, labels_mask, speaker, start_flag]
        theano_vars = [x for x in theano_vars if x is not None]
        theano_vars = list(set(theano_vars))
        theano_vars = {x.name: x for x in theano_vars}

        theano_inputs = []
        numpy_inputs = []

        for key in data_tr.keys():
            theano_inputs.append(theano_vars[key])
            numpy_inputs.append(data_tr[key])

        return function(
            theano_inputs, [sample_x, k, w, pi, phi, pi_att],
            updates=updates)(*numpy_inputs)
