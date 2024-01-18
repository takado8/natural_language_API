import os
from openai import OpenAI

models = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
client = OpenAI(api_key=os.environ['GPT_KEY'])


def generate_audio_file(txt, output_filepath, model):
    print('sending tts request.')
    response = client.audio.speech.create(
        model="tts-1",
        voice=model,
        input=txt
    )
    print('saving to file.')
    response.stream_to_file(output_filepath)


if __name__ == '__main__':
    for model in models:
        for i in range(0, 6):
            generate_audio_file('Erik', f'../data/eryk_tts/eric_tts_{model}{i}.mp3', model)
            exit(5)