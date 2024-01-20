import threading
import librosa.display
import numpy as np
import pyaudio
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from data_processing.whispers import speech_to_txt
from data_processing.mfcc import crop_or_pad
import audioop
from handle_txt import HandleTxtInput
from utils.measure_time import MeasureTime
from collections import deque
import matplotlib.pyplot as plt




recording_time_multiplier = 2
input_length_seconds = 1
# target_sample_rate = 44100
target_sample_rate = 16000
batch_size = 8
frames_per_buffer = 1024
labels_dir = 'data/30 words'
silence_length = 35
output_filename = 'data/temp/audio_record.mp3'
keywords = {'marvin', 'sheila'}


def plot_list(y_values):
    x_values = range(1, len(y_values) + 1)  # Assuming x-values are 1, 2, 3, ...

    plt.plot(x_values, y_values, marker='o')
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Plot of Y-values')
    plt.grid(True)
    plt.show()

class StreamRecognition:
    def __init__(self, model_path='neural_network/30words.h5'):
        print('loading model...')
        self.model_path = model_path
        self.model = None
        self.labels = {0: 'bed', 1: 'bird', 2: 'cat', 3: 'dog', 4: 'down', 5: 'eight', 6: 'five', 7: 'four', 8: 'go', 9: 'happy', 10: 'house', 11: 'left', 12: 'marvin', 13: 'nine', 14: 'no', 15: 'off', 16: 'on', 17: 'one', 18: 'right', 19: 'seven', 20: 'sheila', 21: 'six', 22: 'stop', 23: 'three', 24: 'tree', 25: 'two', 26: 'up', 27: 'wow', 28: 'yes', 29: 'zero'}

        self.previous_ending_frames = None
        self.is_recording_enabled = False
        self.threshold_percent = 0.1
        self.txt_input_handler = HandleTxtInput()

    def recognize(self, frames):
        if not self.model:
            from keras.models import load_model
            self.model = load_model(self.model_path)
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
            percent = int(round(result[prediction] * 100, 0))
            label = self.labels[prediction]
            if not self.is_recording_enabled:
                # print(f'{percent}% {label} ')
                if label in keywords and percent > 95:
                    print('Keyword detected!')
                    self.is_recording_enabled = True
                    self.previous_ending_frames.clear()

    def stream_recognition_async(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
            channels=1,
            rate=target_sample_rate,
            input=True,
            frames_per_buffer=frames_per_buffer)
        try:
            previous_frames = []
            recorded_frames = []
            silent_chunks = 0
            while True:
                frames = []
                volume_list = []
                for i in range(0, int(target_sample_rate / frames_per_buffer * input_length_seconds *
                                      recording_time_multiplier)):
                    data = stream.read(frames_per_buffer)
                    frames.append(data)
                    if self.is_recording_enabled:
                        print('Recording...')
                        rms = audioop.rms(data, 2)  # Get RMS value to determine volume
                        current_volume = rms / 32768  # Normalized RMS as a volume ratio
                        volume_list.append(current_volume)
                        avg_volume = sum(volume_list) / len(volume_list)
                        if avg_volume < self.threshold_percent:
                            # Finish recording when volume drops
                            silent_chunks += 1
                            if silent_chunks >= silence_length:
                                silent_chunks = 0
                                print("Recording stopped due to low volume...")
                                self.is_recording_enabled = False
                        else:
                            silent_chunks = 0
                if self.is_recording_enabled:
                    recorded_frames.extend(frames)

                if recorded_frames and not self.is_recording_enabled:
                    self.save_recording_as_mp3(previous_frames + recorded_frames, p)
                    MeasureTime.start_measure_function_time('speech to text')
                    txt = speech_to_txt(output_filename)
                    execution_time2 = MeasureTime.stop_measure_function_time('speech to text')
                    self.txt_input_handler.handle_txt_input(txt)
                    recorded_frames.clear()
                    print(f'Speech to txt time: {execution_time2}')
                if not self.is_recording_enabled:
                    previous_frames = frames
                    # Offload processing (recognition) to another thread
                    threading.Thread(target=self.recognize, args=(frames,)).start()

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def save_recording_as_mp3(self, frames, p, threshold=0.3):
        # Convert `frames` into an AudioSegment object
        audio_len = len(frames)
        queue_len = 3
        volume_list = deque(maxlen=queue_len)
        avg_list =[]
        silent_chunks = 0
        loud_chunks = 0
        max_silent_chunks = 2
        max_loud_chunks = 2
        i = 0
        top_avg_volume = -1
        low_avg_volume = 999
        top_found = False
        total_avg_volume = sum([audioop.rms(frame, 2) for frame in frames]) / len(frames)
        for data in frames:
            rms = audioop.rms(data, 2)  # Get RMS value to determine volume
            current_volume = rms# / 32768  # Normalized RMS as a volume ratio
            volume_list.append(current_volume)
            if len(volume_list) >= queue_len:
                avg_volume = sum(volume_list) / len(volume_list)
                avg_list.append(avg_volume)

                print(f'avg volume: {avg_volume}')

                # find first word top volume
                if not top_found and avg_volume > top_avg_volume and avg_volume > threshold * total_avg_volume:
                    top_avg_volume = avg_volume

                if not top_found and current_volume < top_avg_volume:
                    silent_chunks += 1
                    if silent_chunks >= max_silent_chunks:
                        top_found = True
                else:
                    silent_chunks = 0
                # find end silence after first word
                if top_found:
                    if avg_volume < low_avg_volume:
                        low_avg_volume = avg_volume

                    if current_volume > avg_volume and current_volume > threshold * avg_volume:
                        loud_chunks += 1
                        if loud_chunks >= max_loud_chunks:
                            # beginning of second word found
                            i -= 2
                            break
                    else:
                        loud_chunks = 0

            i += 1
        # plot_list(avg_list)
        frames = frames[i:]
        cropped_audio_len = len(frames)
        percent_cropped = round((1 - (cropped_audio_len / audio_len)) * 100, 2)
        print(f'cropped: {audio_len} to {cropped_audio_len} by {percent_cropped}%')
        raw_data = b''.join(frames)
        audio_segment = AudioSegment(data=raw_data, sample_width=p.get_sample_size(pyaudio.paInt16),
            frame_rate=target_sample_rate, channels=1)
        # Export the AudioSegment object to an MP3 file
        audio_segment.export(output_filename, format="mp3")
        print(f"Recording has been saved to {output_filename}")


if __name__ == '__main__':
    sr = StreamRecognition()
    sr.stream_recognition_async()