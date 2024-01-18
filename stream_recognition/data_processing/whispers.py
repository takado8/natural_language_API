from openai import OpenAI
import os


client = OpenAI(api_key=os.environ['GPT_KEY'])


def speech_to_txt(filepath):
    audio_file = open(filepath, "rb")
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            language='pl',
            temperature=0.1
        )
    except Exception as ex:
        print(ex)
        return ''
    finally:
        audio_file.close()
    return transcript.lower().replace(".", "").replace("\n", "")


if __name__ == '__main__':
    for file in os.listdir("../data/eryk"):
        txt = speech_to_txt(f"../data/eryk/{file}")
        print(f'{file}: {txt}')
