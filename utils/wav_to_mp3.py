from pydub import AudioSegment

# AudioSegment.ffmpeg = 'C:/ProgramData/chocolatey/lib/ffmpeg/tools/ffmpeg/bin'


def wav_to_mp3(input_path, output_path):
    audio = AudioSegment.from_wav(input_path)
    audio.export(output_path, format="mp3")
