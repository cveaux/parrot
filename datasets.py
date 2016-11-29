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

        assert voice in ['arctic', 'blizzard', 'vctk']

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
        labels_type='full_labels'):

    assert labels_type in [
        'full_labels', 'phonemes', 'unconditional',
        'unaligned_phonemes', 'text']

    dataset = VoiceData(voice=voice, which_sets=which_sets)

    sorting_size = batch_size * sorting_mult

    if not num_examples:
        num_examples = dataset.num_examples

    if 'train' in which_sets:
        scheme = ShuffledExampleScheme(num_examples)
    else:
        scheme = SequentialExampleScheme(num_examples)

    data_stream = DataStream.default_stream(dataset, iteration_scheme=scheme)

    segment_sources = ('features', 'features_mask')
    all_sources = segment_sources

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
    data_stream = parrot_stream('arctic')
    print next(data_stream.get_epoch_iterator())[-1]
    # import ipdb; ipdb.set_trace()
    # epoch_iterator = data_stream.get_epoch_iterator()

    # new_batch = next(epoch_iterator)
    # print '-----  New batch -----'
    # print "features: ", new_batch[0].shape
    # print "labels: ", new_batch[2].shape
    # print "start_flag: ", new_batch[-1]
