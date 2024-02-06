import librosa
from keras.models import load_model
import numpy as np
from stream_recognition.data_processing.mfcc import crop_or_pad


recording_time_multiplier = 2
input_length_seconds = 1
target_sample_rate = 16000
batch_size = 8
keywords = {'marvin'}


class KeywordRecognition:
    def __init__(self, model_path='../neural_network/models/30words2_2.h5'):
        print('loading model...')
        self.model = load_model(model_path)
        print(self.model.summary())
        self.labels = {0: 'bed', 1: 'bird', 2: 'cat', 3: 'dog', 4: 'down', 5: 'eight', 6: 'five', 7: 'four', 8: 'go',
                       9: 'happy', 10: 'house', 11: 'left', 12: 'marvin', 13: 'nine', 14: 'no', 15: 'off', 16: 'on',
                       17: 'one', 18: 'right', 19: 'seven', 20: 'sheila', 21: 'six', 22: 'stop', 23: 'three',
                       24: 'tree', 25: 'two', 26: 'up', 27: 'wow', 28: 'yes', 29: 'zero'}

        self.previous_ending_frames = None

    def recognize(self, frames):
        desired_length_in_samples = int(input_length_seconds * target_sample_rate)
        frame_size = int(len(frames) / recording_time_multiplier)
        frame_shift = int(len(frames) / batch_size)
        mfccs_batch = []
        ending_frames = frames[-int(len(frames)/recording_time_multiplier):]
        if self.previous_ending_frames:
            frames = self.previous_ending_frames + frames
            frames_nb = int(batch_size + batch_size / recording_time_multiplier)
        else:
            frames_nb = batch_size

        for i in range(frames_nb):
            # Convert frames to numpy array and normalize to floating-point
            audio_data = np.frombuffer(b''.join(frames[i * frame_shift:i * frame_shift + frame_size]),
                dtype=np.int16).astype(np.float32) / 32768.0
            audio_data = crop_or_pad(audio_data, desired_length_in_samples)

            mfccs = librosa.feature.mfcc(y=audio_data, sr=target_sample_rate, n_mfcc=13)
            mfccs_batch.append(mfccs)
        self.previous_ending_frames = ending_frames
        # Convert list of mfccs to a numpy array for batch prediction
        x_batch = np.stack(mfccs_batch, axis=0)  # Shape: (snapshot_count, time_steps, num_mfcc)
        x_batch = np.expand_dims(x_batch, axis=-1)  # Add a channel dimension

        # Perform batch prediction
        results = self.model.predict(x_batch, verbose=0, batch_size=len(x_batch))

        for result in results:
            prediction = np.argmax(result)
            percent = round(result[prediction] * 100, 4)
            label = self.labels[prediction]
            if label in keywords and percent > 99.5:
                print(f'{percent} Keyword detected!')
                self.previous_ending_frames.clear()
                return True

        return False
