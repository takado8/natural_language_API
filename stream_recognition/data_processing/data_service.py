import numpy as np

from data_processing.mfcc import generate_mfcc
from data_processing.prepare_training_data import prepare_training_data
from sklearn.model_selection import train_test_split


def load_data(dataset_path, test_size=0.2, length_seconds=0.7, use_augmentation=False):
    features, labels = prepare_training_data(length_seconds=length_seconds, dataset_path=dataset_path,
    use_augmentation=use_augmentation)

    mfccs_train, mfccs_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=test_size)

    # Reshape the data to include the channel dimension
    x_train = mfccs_train[..., np.newaxis]
    x_test = mfccs_test[..., np.newaxis]
    return x_train, x_test, labels_train, labels_test


def load_input_from_file(filepath,length_seconds):
    mfccs = generate_mfcc(filepath, length_seconds)

    # Add both the batch dimension and the channel dimension
    x = np.expand_dims(mfccs, axis=0)
    x = np.expand_dims(x, axis=-1)
    return x


def load_input(mfccs):
    # Add both the batch dimension and the channel dimension
    x = np.expand_dims(mfccs, axis=0)
    x = np.expand_dims(x, axis=-1)
    return x