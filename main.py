from client.gpt_client import GPTClient
from functions.functions_service import FunctionsService
from speech_to_txt.speech_to_txt import SpeechToTxt
import multiprocessing

from utils.measure_time import MeasureTime


KEYWORDS = {'heniu', 'henio'}


def has_keywords(transcript):
    for keyword in KEYWORDS:
        if keyword in transcript:
            return keyword
    return False


if __name__ == '__main__':
    sp = SpeechToTxt()
    functions_service = FunctionsService()
    gpt = GPTClient([f.description for f in functions_service.all_functions.values()])
    text_to_speech = None

    while True:
        transcript = sp.listen()
        print(transcript)
        keyword = has_keywords(transcript.lower())
        if transcript and keyword:
            transcript.replace(keyword, '')
            print("Sending to gpt...")
            MeasureTime.start_measure_function_time('gpt')
            result_queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=gpt.function_call, args=(transcript, result_queue))
            p.start()

            p.join(20)
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

