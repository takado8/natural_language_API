import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import librosa.display


# Load the WAV file
# sampling_rate, data = wavfile.read('speech.wav')
def plot_spectrogram(filepath):
    data, sampling_rate = librosa.load(filepath, sr=None)
    # Make sure the audio file is Mono by taking the first channel if it's Stereo
    if len(data.shape) > 1:
        data = data[:, 0]

    # Use spectrogram function from scipy.signal to generate the spectrogram
    frequencies, times, Sxx = spectrogram(data, fs=sampling_rate)

    # Plot the spectrogram
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()


if __name__ == '__main__':
    import os
    for file in os.listdir("../data/eryk"):
        plot_spectrogram(f"../data/eryk/{file}")
