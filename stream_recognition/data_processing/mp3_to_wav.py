from pydub import AudioSegment

# AudioSegment.ffmpeg = 'C:/ProgramData/chocolatey/lib/ffmpeg/tools/ffmpeg/bin'


def mp3_to_wav(input_path, output_path):
    audio = AudioSegment.from_mp3(input_path)
    audio.export(output_path, format="wav")


if __name__ == '__main__':
    mp3_file_path = 'speech.mp3'
    wav_output_path = 'speech.wav'
    mp3_to_wav(mp3_file_path, wav_output_path)
