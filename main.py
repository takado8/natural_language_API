from client.gpt_client import GPTClient, GPT3, GPT4
from functions.functions_service import FunctionsService
from speech_to_txt.speech_to_txt import SpeechToTxt
import multiprocessing
import os
import json
from utils.measure_time import MeasureTime
from utils.wav_to_mp3 import wav_to_mp3


KEYWORDS = {'eryk', 'eryka', 'eryku'}
KEYWORD_GPT4 = {'turbo'}


def has_keywords(transcript, keywords):
    for keyword in keywords:
        if keyword in transcript:
            return keyword
    return False


def remove_keyword(txt, keyword):
    words = txt.split()
    if keyword in words:
        words.remove(keyword)
        txt = ' '.join(words)
    return txt


def save_audio_file():
    temp_file_path_wav = "data/temp.wav"
    temp_file_path_mp3 = "data/temp.mp3"
    with open("data/index.json") as f:
        index = json.load(f).get("index", None)
    wav_to_mp3(temp_file_path_wav, temp_file_path_mp3)
    os.rename(temp_file_path_mp3, f"data/eryk{index}.mp3")
    index += 1
    with open("data/index.json", 'w') as f:
        json.dump({"index": index}, f)


if __name__ == '__main__':
    sp = SpeechToTxt()
    functions_service = FunctionsService()
    gpt = GPTClient([f.description for f in functions_service.all_functions.values()])
    text_to_speech = None

    while True:
        transcript = sp.listen()
        print(transcript)
        keyword = has_keywords(transcript.lower(), KEYWORDS)
        if transcript and keyword:
            save_audio_file()
            transcript = remove_keyword(transcript, keyword)
            gpt4_keyword = has_keywords(transcript, KEYWORD_GPT4)
            if gpt4_keyword:
                transcript = remove_keyword(transcript, gpt4_keyword)
                gpt_version = GPT4
            else:
                gpt_version = GPT3

            print(f"Sending to {gpt_version}")
            MeasureTime.start_measure_function_time('gpt')
            result_queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=gpt.function_call, args=(transcript, result_queue, gpt_version))
            p.start()

            p.join(30)
            if p.is_alive():
                p.terminate()
                p.join()
            time_consumed_gpt = MeasureTime.stop_measure_function_time('gpt')
            print(f'time consumed gpt: {time_consumed_gpt}')
            if not result_queue.empty():
                gpt_result = result_queue.get()
                print("Result from the process: ")
                print(gpt_result)
            else:
                print("Process did not provide a result within the time limit.")
                continue

            if gpt_result:
                try:
                    function_name = gpt_result['name']
                except:
                    function_name = None
                if function_name:
                    if function_name in functions_service.all_functions:
                        args = gpt_result['args']
                        functions_service.all_functions[function_name](args)
                    else:
                        print(f'Function \'{function_name}\' not found. Available functions:'
                              f' {functions_service.all_functions.keys()}')
                else:
                    try:
                        content = gpt_result['content']
                    except:
                        content = "No content."
                    print(f'\n{content}\n')
                    MeasureTime.start_measure_function_time('tts')
                    if not text_to_speech:
                        from txt_to_speech.txt_to_speech import TxtToSpeech
                        text_to_speech = TxtToSpeech()

                    text_to_speech.speak(content)

