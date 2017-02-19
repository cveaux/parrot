import os
from generate import generate_wav
import numpy
import scipy.io.wavfile
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPTK_DIR = os.environ['SPTK_DIR']
WORLD_DIR = os.environ['WORLD_DIR']

NORM_FILE = 'norm_info_mgc_lf0_vuv_bap_63_MVN.dat'

def vocoder2wav(vocoder_frames, path_to_save=".", name_prefix="samples", dataset="vctk", post_filter=False):
    """
    Returns a list of raw audio and a list of (sampled_frequencies, time, specgram_values)
    """
    assert(numpy.abs(vocoder_frames).max() < 20.), "Make sure that range of frame values is small"

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    raw_audio = []
    audio_spec = []

    for i, this_sample in enumerate(vocoder_frames):
        generate_wav(
            this_sample,
            path_to_save,
            name_prefix + '_' + str(i),
            sptk_dir=SPTK_DIR,
            world_dir=WORLD_DIR,
            norm_info_file=os.path.join(os.environ['FUEL_DATA_PATH'], dataset, NORM_FILE),
            do_post_filtering=post_filter)

        rate, data = scipy.io.wavfile.read(os.path.join(path_to_save, name_prefix + '_' + str(i)+'.wav'))
        raw_audio.append(data)
        f, t, Sxx = signal.spectrogram(data, rate)
        audio_spec.append((f, t, Sxx))

    return raw_audio, audio_spec

def get_sample_raw_and_spec():
    """
    Example taken from the scipy website: 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    """
    fs = 10e3
    N = 1e5
    amp = 2 * numpy.sqrt(2)
    noise_power = 0.001 * fs / 2
    time = numpy.arange(N) / fs
    freq = numpy.linspace(1e3, 2e3, N)
    x = amp * numpy.sin(2*numpy.pi*freq*time)
    x += numpy.random.normal(scale=numpy.sqrt(noise_power), size=time.shape)
    scipy.io.wavfile.write(
                    os.path.join('test_wav', "spec_test.wav"),
                    fs,
                    x)

    rate, data = scipy.io.wavfile.read(os.path.join('test_wav', "spec_test.wav"))
    f, t, Sxx = signal.spectrogram(data, rate)  
    return data, (f, t, Sxx)


def test_vocoder2wav():
    frames = numpy.random.uniform(size=(2, 10, 63))
    raw_audio, audio_spec = vocoder2wav(frames, 'test_wav')
    for i in range(len(raw_audio)):
        plt.plot(numpy.arange(len(raw_audio[i])), raw_audio[i])
        plt.savefig(os.path.join('test_wav', 'sample_temporal_{}.png'.format(i)))
        plt.close()
        plt.pcolormesh(audio_spec[i][1], audio_spec[i][0], audio_spec[i][2])
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time')
        plt.savefig(os.path.join('test_wav', 'sample_specgram_{}.png'.format(i)))
        plt.close()

def test_plot():
    raw_data, specgram = get_sample_raw_and_spec()
    plt.plot(numpy.arange(len(raw_data)), raw_data)
    plt.savefig(os.path.join('test_wav', 'sample_test_raw.png'))
    plt.close()
    plt.pcolormesh(specgram[1], specgram[0], specgram[2])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time')
    plt.savefig(os.path.join('test_wav', 'sample_test_specgram.png'))
    plt.close()

if __name__ == "__main__":
    test_plot()
    test_vocoder2wav()
