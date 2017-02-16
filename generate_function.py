import os
from generate import generate_wav
import numpy

SPTK_DIR = os.environ['SPTK_DIR']
WORLD_DIR = os.environ['WORLD_DIR']

NORM_FILE = 'norm_info_mgc_lf0_vuv_bap_63_MVN.dat'

def vocoder2wav(vocoder_frames, path_to_save=".", name_prefix="samples", dataset="vctk", post_filter=False):
    assert(numpy.abs(vocoder_frames).max() < 20.), "Make sure that range of frame values is small"

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    for i, this_sample in enumerate(vocoder_frames):
        generate_wav(
            this_sample,
            path_to_save,
            name_prefix + '_' + str(i),
            sptk_dir=SPTK_DIR,
            world_dir=WORLD_DIR,
            norm_info_file=os.path.join(os.environ['FUEL_DATA_PATH'], dataset, NORM_FILE),
            do_post_filtering=post_filter)


def test_vocoder2wav():
    frames = numpy.random.uniform(size=(10, 10, 63))
    vocoder2wav(frames, 'test_wav')

if __name__ == "__main__":
    test_vocoder2wav()
