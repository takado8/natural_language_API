import os

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import tensorflow as tf
from ..data_processing.data_service import load_data, load_input_from_file
from keras.models import load_model, save_model

from ..data_processing.mfcc import generate_mfcc


gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs: {gpus}')


def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
    model.add(BatchNormalization())

    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))
    return model


def compile_and_fit(model, x_train, x_test, y_train, y_test, epochs=15, batch_size=3):
    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return model


def predict_directory(directory, model_path, length_seconds, segregate=False):
    model = load_model(model_path)
    # nb_of_positives = os.listdir(f'{directory}/{0}')
    correct = 0
    incorrect = 0
    moved_to_0 = 0
    moved_to_1 = 0
    for i in range(len(os.listdir(directory))):
        dirpath = f'{directory}/{i}'
        for file in os.listdir(dirpath):
            x = load_input_from_file(filepath=f'{dirpath}/{file}', length_seconds=length_seconds)
            result = model.predict(x, verbose=0)[0]
            prediction = np.argmax(result)
            percent = round(result[prediction] * 100, 2)
            # if prediction == 1 and percent < 95:
            #     prediction = 0
            is_correct = prediction == i
            if is_correct:
                correct += 1
            else:
                incorrect += 1
                if segregate and percent > 60:
                    not_i = int(not bool(i))
                    src = f'{dirpath}/{file}'
                    target = f'{directory}/{not_i}/{file}'
                    try:
                        os.rename(src, target)
                        if not_i:
                            moved_to_1 += 1
                        else:
                            moved_to_0 += 1
                        print(f'{moved_to_0 + moved_to_1}. moving {file} to {target}')
                    except FileExistsError:
                        print(f'file exists {file}')
                        os.remove(src)
            arrow = '<<<<<<<<<' if not is_correct else ''
            print(f'{file}: {percent}% {is_correct}{arrow}')
    print(f'correct: {correct}\nincorrect: {incorrect}'
          f'\naccuracy: {round(correct / (correct + incorrect) * 100, 2)}%')
    print(f'moved to 0: {moved_to_0}\nmoved to 1: {moved_to_1}')


def segregate_directory(directory, model_path, length_seconds=0.5):
    model = load_model(model_path)
    output_shape = model.layers[-1].output_shape[-1]
    print(f'Output shape: {output_shape}')
    files = os.listdir(directory)
    n = len(files)
    print(f'Found {n} files.')
    results = {}
    for i in range(output_shape):
        subdir = f'{directory}/{i}'
        if not os.path.isdir(subdir):
            os.mkdir(subdir)
            print(f'Directory created: {subdir}')

    i = 0
    for file in files:
        i += 1
        x = load_input_from_file(filepath=f'{directory}/{file}', length_seconds=length_seconds)
        result = model.predict(x, verbose=0)[0]
        prediction = np.argmax(result)
        percent = round(result[prediction] * 100, 2)

        if percent > 95:
            src = f'{directory}/{file}'
            target = f'{directory}/{prediction}/{file}'
            try:
                os.rename(src, target)
                print(f'{i}/{n}. Moving {file} to {target}')
                if prediction in results:
                    results[prediction] += 1
                else:
                    results[prediction] = 1
            except FileExistsError:
                print(f'File exists {target}. Removing source.')
                os.remove(src)
    for label in results:
        print(f'{label}: {results[label]}')
    unmatched = n - sum(results.values())
    print(f'Unmatched: {unmatched}')


def save_inputs(directory, x, x_test, y, y_test):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    np.save(f'{directory}/x_train.npy', x)
    np.save(f'{directory}/x_test.npy', x_test)
    np.save(f'{directory}/labels_train.npy', y)
    np.save(f'{directory}/labels_test.npy', y_test)


def load_inputs(directory):
    x = np.load(f'{directory}/x_train.npy')
    x_test = np.load(f'{directory}/x_test.npy')
    y = np.load(f'{directory}/labels_train.npy')
    y_test = np.load(f'{directory}/labels_test.npy')
    return x, x_test, y, y_test


if __name__ == '__main__':
    inputs_dir = '../data/30words_inputs'

    # x_train, x_test, labels_train, labels_test = (
    #     load_data('../data/30 words', 0.15, length_seconds=1,
    #         use_augmentation=True))
    # save_inputs(inputs_dir, x_train, x_test, labels_train, labels_test)

    x_train, x_test, labels_train, labels_test = load_inputs(inputs_dir)

    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    output_shape = labels_train.shape[1]
    model = create_model(input_shape, output_shape)
    model.summary()
    model = compile_and_fit(model, x_train, x_test, labels_train, labels_test, epochs=12, batch_size=32)
    save_model(model, '30words2.h5')
    #
    # predict_directory(directory='../data/eryk_training_2',
    #     model_path='eryk_newnet3.h5', length_seconds=0.5, segregate=False)

    # segregate_directory(directory='../data/segregation', model_path='eryk500_noise7.h5')
    # model = load_model('eryk_newnet3.h5')
    # model.summary()
