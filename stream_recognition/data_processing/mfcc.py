import matplotlib.pyplot as plt
import librosa.display
import numpy as np


# target_sample_rate = 44100
target_sample_rate = 16000


def crop_or_pad(signal, samples_length):
    # Crop or pad the signal
    if len(signal) > samples_length:
        # Crop the signal if it's longer than the desired length
        signal = signal[:samples_length]
    elif len(signal) < samples_length:
        # Pad with zeros if the signal is shorter than the desired length
        padding = samples_length - len(signal)
        signal = np.pad(signal, (0, padding), 'constant')
    return signal


def generate_mfcc(filepath, length_seconds):
    signal, sample_rate = librosa.load(filepath, sr=None)
    if sample_rate != target_sample_rate:
        print(f'changing sample rate: {sample_rate} to {target_sample_rate}')
        signal = librosa.resample(y=signal, orig_sr=sample_rate, target_sr=target_sample_rate)
    desired_length_in_samples = int(length_seconds * target_sample_rate)
    signal = crop_or_pad(signal, desired_length_in_samples)
    mfccs = librosa.feature.mfcc(y=signal, sr=target_sample_rate, n_mfcc=13)
    return mfccs


def generate_augmented_mfccs(filepath, length_seconds):
    signal, sample_rate = librosa.load(filepath, sr=None)
    if sample_rate != target_sample_rate:
        signal = librosa.resample(y=signal, orig_sr=sample_rate, target_sr=target_sample_rate)
    desired_length_in_samples = int(length_seconds * target_sample_rate)
    signal = crop_or_pad(signal, desired_length_in_samples)
    mfccs = librosa.feature.mfcc(y=signal, sr=target_sample_rate, n_mfcc=13)

    samples = []
    for i in range(2):
        # # Time-stretching
        ts_0_9 = crop_or_pad(librosa.effects.time_stretch(signal, rate=0.9 + i * 0.05), desired_length_in_samples)
        ts_1_1 = crop_or_pad(librosa.effects.time_stretch(signal, rate=1.1 - i * 0.05), desired_length_in_samples)

        # Pitch-shifting by two semitones
        ps_down = crop_or_pad(librosa.effects.pitch_shift(signal, sr=target_sample_rate, n_steps=-(i+1)),
            desired_length_in_samples)
        ps_up = crop_or_pad(librosa.effects.pitch_shift(signal, sr=target_sample_rate, n_steps=i+1),
            desired_length_in_samples)
        samples.append(ts_0_9)
        samples.append(ts_1_1)
        samples.append(ps_up)
        samples.append(ps_down)
    augmented_mfccs = [librosa.feature.mfcc(y=y, sr=target_sample_rate, n_mfcc=13) for y in samples]
    augmented_mfccs.append(mfccs)
    return augmented_mfccs


def plot_MFCC(mfcc, sample_rate):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sample_rate)
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import os
    mfccs = []
    for file in os.listdir("../data/eryk"):
        mfccs.append(generate_mfcc(f"../data/eryk/{file}", 1))

    for mfcc in mfccs:
        plot_MFCC(mfcc, target_sample_rate)

