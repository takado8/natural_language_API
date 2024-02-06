import asyncio
import threading
import uuid

import pyaudio
from pydub import AudioSegment
from data_processing.whispers import speech_to_txt
import audioop
from handle_txt import HandleTxtInput
from utils.measure_time import MeasureTime
from collections import deque
import matplotlib.pyplot as plt
from keyword_recognition.call_api import send_frame, get_recognition


recording_time_multiplier = 2
input_length_seconds = 1
# target_sample_rate = 44100
target_sample_rate = 16000
batch_size = 8
frames_per_buffer = 1024
labels_dir = 'data/30 words'
silence_length = 35
output_filename = 'data/temp/audio_record.mp3'
keyword_recording_dir = 'data/temp/marvin'
keywords = {'marvin'}


def plot_list(y_values, line_x=None, lines_y=None):
    x_values = range(1, len(y_values) + 1)  # Assuming x-values are 1, 2, 3, ...

    plt.plot(x_values, y_values, marker='o')

    if line_x is not None:
        plt.axvline(x=line_x, color='r', linestyle='--', label='Cut')

    if lines_y is not None:

        plt.axhline(y=lines_y[0], color='b', linestyle='-.', label=f'Avg vol {lines_y[0]}')
        plt.axhline(y=lines_y[1], color='g', linestyle='-.', label=f'Thresh {lines_y[1]}')

    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Plot of Y-values')
    plt.grid(True)
    plt.legend()  # Add legend if there are lines
    plt.show()


class StreamRecognition:
    def __init__(self, model_path='neural_network/30words2.h5'):
        self.lock = threading.Lock()
        self.is_running = False
        self.model_path = model_path
        self.model = None
        # self.labels = {0: 'none', 1: 'sheila'}

        self.is_recording_enabled = False
        self.threshold_percent = 0.1
        self.txt_input_handler = HandleTxtInput()
        self.sample_size = None

    async def stream_recognition_async(self):
        py_audio = pyaudio.PyAudio()
        if not self.sample_size:
            self.sample_size = py_audio.get_sample_size(pyaudio.paInt16)

        stream = py_audio.open(format=pyaudio.paInt16,
            channels=1,
            rate=target_sample_rate,
            input=True,
            frames_per_buffer=frames_per_buffer)
        try:
            previous_frames = []
            recorded_frames = []
            silent_chunks = 0
            result_task = None
            while True:
                frames = []
                volume_list = []
                for i in range(0, int(target_sample_rate / frames_per_buffer * input_length_seconds *
                                      recording_time_multiplier)):
                    data = stream.read(frames_per_buffer)
                    frames.append(data)
                    send_frame(data)
                    await asyncio.sleep(0)
                    if result_task is not None and result_task.done():
                        result = await result_task
                        print(result)
                        if result == "True":
                            self.is_recording_enabled = True
                        result_task = None
                    # else:
                    #     print(result_task)
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
                    cropped_frames, keyword_frames = self.crop_keyword_from_recording(previous_frames + recorded_frames)
                    self.save_as_mp3(cropped_frames)
                    self.save_as_mp3(keyword_frames,
                        filename=f'{keyword_recording_dir}/marvin{str(uuid.uuid4())[-8:]}.mp3')
                    MeasureTime.start_measure_function_time('speech to text')
                    txt = speech_to_txt(output_filename)
                    # txt = ''
                    execution_time2 = MeasureTime.stop_measure_function_time('speech to text')
                    self.txt_input_handler.handle_txt_input(txt)
                    recorded_frames.clear()
                    print(f'Speech to txt time: {execution_time2}')
                if not self.is_recording_enabled and result_task is None:
                    previous_frames = frames
                    print('creating task')
                    result_task = asyncio.create_task(get_recognition())

        finally:
            stream.stop_stream()
            stream.close()
            py_audio.terminate()

    def crop_keyword_from_recording(self, frames, threshold=0.9):
        audio_len = len(frames)
        queue_len = 3
        volume_list = deque(maxlen=queue_len)
        avg_list = []
        silent_chunks = 0
        loud_chunks = 0
        max_silent_chunks = 3
        max_loud_chunks = 2
        i = 0
        top_avg_volume = -1
        low_avg_volume = 999
        top_found = False
        low_found = False
        low_point = -1
        total_avg_volume = sum([audioop.rms(frame, 2) for frame in frames]) / len(frames)
        for data in frames:
            current_volume = audioop.rms(data, 2)  # Get RMS value to determine volume
            volume_list.append(current_volume)
            if len(volume_list) >= queue_len:
                avg_volume = sum(volume_list) / len(volume_list)
                avg_list.append(avg_volume)
                # find first word top volume
                if not low_found:
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

                        if current_volume > avg_volume and current_volume > threshold * total_avg_volume:
                            loud_chunks += 1
                            if loud_chunks >= max_loud_chunks:
                                # beginning of second word found
                                i -= 2
                                low_found = True
                                low_point = i
                        else:
                            loud_chunks = 0
            i += 1
        plot_list(avg_list, low_point, (total_avg_volume, total_avg_volume * threshold))
        main_frames = frames[low_point:]
        cropped_frames = frames[:low_point]
        cropped_audio_len = len(main_frames)
        percent_cropped = round((1 - (cropped_audio_len / audio_len)) * 100, 2)
        print(f'cropped: {audio_len} to {cropped_audio_len} by {percent_cropped}%')
        return main_frames, cropped_frames

    def save_as_mp3(self, frames, filename=output_filename):
        raw_data = b''.join(frames)
        audio_segment = AudioSegment(data=raw_data, sample_width=self.sample_size,
            frame_rate=target_sample_rate, channels=1)
        audio_segment.export(filename, format="mp3")
        print(f"Recording has been saved to {filename}")


if __name__ == '__main__':
    sr = StreamRecognition()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(sr.stream_recognition_async())