import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from stream_recognition.data_processing.mfcc import generate_mfcc, generate_augmented_mfccs


def prepare_training_data(length_seconds, dataset_path="../data", use_augmentation=False):
    labels = []
    features = []
    i = 0
    inputs_per_category = min([len(os.listdir(os.path.join(dataset_path, subdir))) for subdir in
                               os.listdir(dataset_path)])
    # inputs_per_category = 100
    print(f'Inputs per category: {inputs_per_category}')
    for label in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, label)
        print(f'generating MFCCs for {subfolder_path}')
        n = 0
        for file_name in os.listdir(subfolder_path):
            if n >= inputs_per_category:
                break
            split = file_name.split('.')
            if split[-1] not in {'wav', 'mp3'}:
                continue
            n += 1
            file_path = os.path.join(subfolder_path, file_name)

            if use_augmentation:
                mfccs = generate_augmented_mfccs(file_path, length_seconds)
            else:
                mfccs = [generate_mfcc(file_path, length_seconds=length_seconds)]

            if mfccs is not None:
                for mfcc in mfccs:
                    features.append(mfcc)
                    labels.append(i)
        i += 1

    features = np.array(features)
    labels = np.array(labels)
    le = LabelEncoder()
    labels_encoded = to_categorical(le.fit_transform(labels))
    return features, labels_encoded
