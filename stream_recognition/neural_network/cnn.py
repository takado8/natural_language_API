import os

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import tensorflow as tf
from keras import layers, models
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import keras
from stream_recognition.data_processing.data_service import load_data, load_input_from_file
from keras.models import load_model, save_model


print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
# from ..data_processing.mfcc import generate_mfcc
log_dir = "logs/"

gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs: {gpus}')


def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))
    return model


def compile_and_fit(model, x_train, x_test, y_train, y_test, epochs=15, batch_size=3,
                    checkpoints_path=None, tensorboard_log_dir=log_dir):
    callbacks = []
    if checkpoints_path:
        print(f'checkpoints_path: {checkpoints_path}')
        checkpoint_callback = ModelCheckpoint(filepath=checkpoints_path + '_{epoch:02d}.h5', save_freq='epoch')
        callbacks.append(checkpoint_callback)
    if tensorboard_log_dir:
        print(f'tensorboard_log_dir: {tensorboard_log_dir}')
        tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, update_freq=10, write_images=True)
        callbacks.append(tensorboard_callback)

    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
        shuffle=True, callbacks=callbacks)
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return model


def create_fine_tuned_model(pretrained_model_path):
    # Load your existing pretrained model
    pretrained_model = tf.keras.models.load_model(pretrained_model_path)

    # Display the summary of the pretrained model
    pretrained_model.summary()

    # Create new layers for selective fine-tuning
    output_layer = layers.Dense(128, activation='relu', name='dense_fine')(pretrained_model.layers[-2].output)
    dropout_layer = layers.Dropout(0.5, name='dropout_fine')(output_layer)  # Adjust the dropout rate as needed
    final_output = layers.Dense(2, activation='softmax', name='dense_fine2')(dropout_layer)

    # Combine the pretrained model with the new layers
    fine_tuned_model = models.Model(inputs=pretrained_model.input, outputs=final_output)

    # Freeze layers related to the original 30 classes
    for layer in fine_tuned_model.layers[:-5]:
        layer.trainable = False

    return fine_tuned_model


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
    inputs_dir = '../data/30words_inputs_augmented'
    inputs_dir2 = '../data/fine_tuning_inputs'

    # x_train, x_test, labels_train, labels_test = (
    #     load_data('../data/temp/', 0.0001, length_seconds=1,
    #         use_augmentation=True))
    # save_inputs(inputs_dir, x_train, x_test, labels_train, labels_test)

    x_train, x_test, labels_train, labels_test = load_inputs(inputs_dir)
    x_train2, x_test2, labels_train2, labels_test2 = load_inputs(inputs_dir2)
    xtr = np.concatenate((x_train, x_train2), axis=0)
    # xtst = np.concatenate((x_test, x_test2), axis=0)

    # Concatenate labels along the specified axis
    print(labels_train.shape)
    print(labels_train2.shape)
    ytr = np.concatenate((labels_train, labels_train2), axis=0)
    # ytst = np.concatenate((labels_test, labels_test2), axis=0)
    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    output_shape = labels_train.shape[1]

    # model = create_model(input_shape, output_shape)
    model = load_model('30words2.h5')
    model.summary()
    model = compile_and_fit(model, xtr, x_test, ytr, labels_test, epochs=12, batch_size=32,
                        checkpoints_path='checkpoints/30words2_2')
    save_model(model, 'models/30words2_2.h5')

    # predict_directory(directory='../data/eryk_training_2',
    #     model_path='eryk_newnet3.h5', length_seconds=0.5, segregate=False)

    # segregate_directory(directory='../data/segregation', model_path='eryk500_noise7.h5')

    # model.summary()
    # model = create_fine_tuned_model('30words.h5')
    # model.summary()
    # model = compile_and_fit(model, x_train, x_test, labels_train, labels_test, epochs=21, batch_size=1)
    # save_model(model, 'fine_tuned2.h5')
