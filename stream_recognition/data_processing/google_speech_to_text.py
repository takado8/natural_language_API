import speech_recognition as sr
from pydub import AudioSegment
import io

class SpeechToTxt:

    @staticmethod
    def recognize(input_file_path):
        # Load audio file with pydub
        audio = AudioSegment.from_mp3(input_file_path)

        # Convert pydub.AudioSegment to raw audio data
        audio_data = io.BytesIO()
        audio.export(audio_data, format="wav")
        audio_data.seek(0)

        # Use the speech_recognition library to recognize the speech
        text_out = None
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_data) as source:
                audio_recorded = recognizer.record(source)

            print('sending to google...')
            text_out = recognizer.recognize_google(audio_recorded, language="pl-PL", show_all=False)
            print('received: ' + text_out)
        except Exception as e:
            print("Error recognizing audio: " + str(e))
        finally:
            return text_out


if __name__ == '__main__':
    import os
    for file in os.listdir("../data/eryk"):
        txt = SpeechToTxt.recognize(f"../data/eryk/{file}")
        print(f'text: {txt}')