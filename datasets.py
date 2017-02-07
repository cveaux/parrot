import os

from fuel import config
from fuel.schemes import (
    ConstantScheme, ShuffledExampleScheme,
    SequentialExampleScheme)
from fuel.transformers import (
    AgnosticSourcewiseTransformer, Batch, Filter, FilterSources,
    Mapping, Padding, Rename, SortMapping, Transformer, Unpack)
from fuel.streams import DataStream

from fuel.datasets import H5PYDataset

import numpy


def _length(data):
    return len(data[0])


def _transpose(data):
    return data.swapaxes(0, 1)


def _check_batch_size(data, batch_size):
    return len(data[0]) == batch_size


def _check_ratio(data, idx1, idx2, min_val, max_val):
    ratio = len(data[idx1]) / float(len(data[idx2]))
    # print (min_val <= ratio and ratio <= max_val)
    return (min_val <= ratio and ratio <= max_val)


def get_quantizers(limit=5., quantisation='linear', levels=256):

    def quantise(x):
        assert(limit > 0.)
        x[x > limit] = limit
        x[x < -limit] = -limit

        x = x+limit

        if quantisation == 'linear':
            x /= 2*limit
        else:
            raise NotImplementedError("{} quantisation not implemented!!".format(quantisation))

        x = numpy.int32(x*levels - 0.00001)

        return x

    def dequantise(x):
        x = numpy.float32(x + 0.5)/numpy.float32(levels)
        if quantisation == 'linear':
            x = x*2*limit - limit
        else:
            raise NotImplementedError("{} quantisation not implemented!!".format(quantisation))
        return numpy.float32(x)

    return quantise, dequantise



class SegmentSequence(Transformer):
    """Segments the sequences in a batch.

    This transformer is useful to do tbptt. All the sequences to segment
    should have the time dimension as their first dimension.
    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The wrapped data stream.
    seq_size : int
        maximum size of the resulting sequences.
    which_sources : tuple of str, optional
        sequences to segment
    add_flag : bool, optional
        add a flag indicating the beginning of a new sequence.
    flag_name : str, optional
        name of the source for the flag
    min_size : int, optional
        smallest possible sequence length for the last cut
    return_last : bool, optional
        return the last cut of the sequence, which might be different size
    share_value : int, optional
        size of overlap
    """

    def __init__(self,
                 data_stream,
                 seq_size=100,
                 which_sources=None,
                 add_flag=False,
                 flag_name=None,
                 min_size=10,
                 return_last=True,
                 share_value=0,
                 **kwargs):

        super(SegmentSequence, self).__init__(
            data_stream=data_stream,
            produces_examples=data_stream.produces_examples,
            **kwargs)

        if which_sources is None:
            which_sources = data_stream.sources
        self.which_sources = which_sources

        self.seq_size = seq_size
        self.step = 0
        self.data = None
        self.len_data = None
        self.add_flag = add_flag
        self.min_size = min_size
        self.share_value = share_value

        if not return_last:
            self.min_size += self.seq_size

        if flag_name is None:
            flag_name = u"start_flag"

        self.flag_name = flag_name

    @property
    def sources(self):
        return self.data_stream.sources + ((self.flag_name,)
                                           if self.add_flag else ())

    def get_data(self, request=None):
        flag = 0

        if self.data is None:
            self.data = next(self.child_epoch_iterator)
            idx = self.sources.index(self.which_sources[0])
            self.len_data = self.data[idx].shape[0]
            flag = 1  # if flag is here: first part

        segmented_data = list(self.data)

        for source in self.which_sources:
            idx = self.sources.index(source)
            # Segment data:
            segmented_data[idx] = self.data[idx][
                self.step:(self.step + self.seq_size)]

        self.step += self.seq_size

        # Size of overlap:
        self.step -= self.share_value

        if self.step + self.min_size >= self.len_data:
            self.data = None
            self.len_data = None
            self.step = 0
            # flag = 1  # if flag is here: last part

        if self.add_flag:
            segmented_data.append(flag)

        return tuple(segmented_data)

class SourceMapping(AgnosticSourcewiseTransformer):
    """Apply a function to a subset of sources.

    Similar to the Mapping transformer but for a subset of sources.
    It will apply the same function to each source.
    Parameters
    ----------
    mapping : callable

    """

    def __init__(self, data_stream, mapping, **kwargs):
        """Initialization.

        Parameters:
            data_stream: DataStream
            mapping: callable object
        """
        self.mapping = mapping
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(SourceMapping, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)

    def transform_any_source(self, source_data, _):
        return numpy.asarray(self.mapping(source_data))


class AddConstantSource(Mapping):
    def __init__(self, data_stream, constant, name, **kwargs):
        super(AddConstantSource, self).__init__(
            data_stream, lambda x: (constant,), (name,), **kwargs)


class VoiceData(H5PYDataset):
    def __init__(self, voice, which_sets, filename=None, **kwargs):

        assert voice in ['arctic', 'blizzard', 'dimex', 'vctk', 'librispeech']

        self.voice = voice

        if filename is None:
            filename = voice

        self.filename = filename + '.hdf5'
        super(VoiceData, self).__init__(self.data_path, which_sets, **kwargs)

    @property
    def data_path(self):
        return os.path.join(config.data_path[0], self.voice, self.filename)


def parrot_stream(
        voice, use_speaker=False, which_sets=('train',), batch_size=32,
        seq_size=50, num_examples=None, sorting_mult=4, noise_level=None,
        labels_type='full_labels', quantize_features=False, check_ratio=True, raw_audio=False):

    assert labels_type in [
        'full_labels', 'phonemes', 'unconditional',
        'unaligned_phonemes', 'text']

    dataset = VoiceData(voice=voice, which_sets=which_sets)

    sorting_size = batch_size * sorting_mult

    if not num_examples:
        num_examples = dataset.num_examples

    # print num_examples

    if 'train' in which_sets:
        scheme = ShuffledExampleScheme(num_examples)
    else:
        scheme = SequentialExampleScheme(num_examples)

    data_stream = DataStream.default_stream(dataset, iteration_scheme=scheme)

    # print data_stream.sources

    if check_ratio and labels_type in ['unaligned_phonemes', 'text']:
        idx = data_stream.sources.index(labels_type)
        min_val = 8 if labels_type == 'text' else 12.
        max_val = 16 if labels_type == 'text' else 25.
        data_stream = Filter(
            data_stream, lambda x: _check_ratio(x, 0, idx, min_val, max_val))

    segment_sources = ('features', 'features_mask')
    all_sources = segment_sources

    if raw_audio:
        all_sources += ('raw_audio', )

    if labels_type != 'unconditional':
        all_sources += ('labels', )
        data_stream = Rename(data_stream, {labels_type: 'labels'})

    if labels_type in ['full_labels', 'phonemes']:
        segment_sources += ('labels',)

    elif labels_type in ['unaligned_phonemes', 'text']:
        all_sources += ('labels_mask', )

    data_stream = Batch(
        data_stream, iteration_scheme=ConstantScheme(sorting_size))
    data_stream = Mapping(data_stream, SortMapping(_length))
    data_stream = Unpack(data_stream)
    data_stream = Batch(
        data_stream, iteration_scheme=ConstantScheme(batch_size))

    data_stream = Filter(
        data_stream, lambda x: _check_batch_size(x, batch_size))

    data_stream = Padding(data_stream)

    if use_speaker:
        data_stream = FilterSources(
            data_stream, all_sources + ('speaker_index',))
    else:
        data_stream = FilterSources(
            data_stream, all_sources)

    data_stream = SourceMapping(
        data_stream, _transpose, which_sources=segment_sources)

    if quantize_features:
        quantize, _ = get_quantizers()
        data_stream = SourceMapping(
            data_stream, quantize, which_sources='features')

    data_stream = SegmentSequence(
        data_stream,
        seq_size=seq_size + 1,
        share_value=1,
        return_last=False,
        add_flag=True,
        which_sources=segment_sources)

    if noise_level is not None:
        data_stream = AddConstantSource(
            data_stream, noise_level, 'feedback_noise_level')

    return data_stream


if __name__ == "__main__":
    data_stream = parrot_stream('dimex', labels_type='text', seq_size=20,batch_size=5, raw_audio=True)
    print data_stream.sources
    # exit()
    import ipdb; ipdb.set_trace()


    # print next(data_stream.get_epoch_iterator())[-1]

    # For Arctic, the ratio is 18 steps of features per letter.
    # data_tr = next(data_stream.get_epoch_iterator())
    # ratios = (data_tr[1].sum(0) / data_tr[3].sum(1))
    # print numpy.percentile(ratios, [0, 10, 25, 50, 75, 90, 99, 100])

    # Arctic
    # phonemes: array([ 12.84, 14.75, 15.56, 16.82, 18.16, 19.89, 48.8])
    # text:     array([  8.2, 9.89, 10.39, 11.07, 11.91, 12.81, 24.4])

    # Blizzard
    # phonemes: array([ 6.26, 14.07, 15.11, 16.26, 17.60, 19.23, 103.33])
    # text:     array([4.37, 9.8, 10.64, 11.62, 12.59, 13.76, 46. ])

    # VCTK
    # phonemes: array([  3., 12.39, 13.52, 15.03, 16.8, 18.96, 40.5])
    # text:     array([  2.04, 8.43, 9.23, 10.28, 11.56, 13.03, 23.15])

    ep_iter = data_stream.get_epoch_iterator()
    try:
        data_batch = next(ep_iter)
        features_reshaped = data_batch[0].reshape((-1, 63))
        data_mask = data_batch[1].flatten()
        features_useful = features_reshaped[data_mask == 1.]

        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        means = features_useful.mean(axis=0)
        stds = features_useful.std(axis=0)
        # n, bins, patches = plt.hist(features_useful.flatten(), 256)
        # plt.plot(bins, n, 'r--', linewidth=1)
        plt.plot(numpy.arange(len(means)), means, '.r')
        plt.plot(numpy.arange(len(stds)), stds, '.b')
        for i in range(len(means)):
            count_i = len(numpy.unique(features_useful[:,i]))
            if count_i > 5:
                count_i = 5.
            plt.plot(i, count_i, '.g')
        plt.savefig("data_dim_dist.jpg")
    except StopIteration:
        print "Done processing data iterator"

    # ep_iter = data_stream.get_epoch_iterator()
    # data_batch = next(ep_iter)
    # SPTK_DIR = '/data/lisatmp4/kumarkun/merlin/tools/bin/SPTK-3.9/'
    # WORLD_DIR = '/data/lisatmp4/kumarkun/merlin/tools/bin/WORLD/'

    # norm_info_file = os.path.join(
    #     os.environ['FUEL_DATA_PATH'], 'vctk',
    #     'norm_info_mgc_lf0_vuv_bap_63_MVN.dat')

    # if not os.path.exists(os.path.join(os.environ['RESULTS_DIR'], 'vctk', 'actual_samples')):
    #     os.makedirs(os.path.join(os.environ['RESULTS_DIR'], 'vctk', 'actual_samples'))

    # if not os.path.exists(os.path.join(os.environ['RESULTS_DIR'], 'vctk', 'normalisation_reconstructed')):
    #     os.makedirs(os.path.join(os.environ['RESULTS_DIR'], 'vctk', 'normalisation_reconstructed'))


    # from generate import generate_wav
    # normalise, denormalise = get_quantizers(5)

    # for i, this_sample in enumerate(data_batch[0].transpose(1,0,2)):
    #     this_sample = this_sample[:int(data_batch[1].sum(axis=0)[i])]

    #     normalised_reconstructed_this = denormalise(normalise(this_sample))

    #     generate_wav(
    #         this_sample,
    #         os.path.join(os.environ['RESULTS_DIR'], 'vctk', 'actual_samples'),
    #         'sample' + '_' + str(i),
    #         sptk_dir=SPTK_DIR,
    #         world_dir=WORLD_DIR,
    #         norm_info_file=norm_info_file,
    #         do_post_filtering=False)

    #     generate_wav(
    #         normalised_reconstructed_this,
    #         os.path.join(os.environ['RESULTS_DIR'], 'vctk', 'normalisation_reconstructed'),
    #         'sample' + '_' + str(i),
    #         sptk_dir=SPTK_DIR,
    #         world_dir=WORLD_DIR,
    #         norm_info_file=norm_info_file,
    #         do_post_filtering=False)
