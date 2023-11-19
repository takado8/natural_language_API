import os
from openai import OpenAI

from utils.measure_time import MeasureTime
from .sound_player import play_sound


class TxtToSpeech:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ['GPT_KEY'])
        self.speech_file_path = "speech.mp3"

    def speak(self, txt):
        print('sending tts request.')
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=txt
        )
        print('saving to file.')
        response.stream_to_file(self.speech_file_path)
        print('playing sound.')
        tts_time_consumed = MeasureTime.stop_measure_function_time('tts')
        print(f'time consumed tts: {tts_time_consumed}')
        play_sound(self.speech_file_path)
