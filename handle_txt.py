from client.gpt_client import GPTClient, GPT3, GPT4
from functions.functions_service import FunctionsService
import multiprocessing

from utils.measure_time import MeasureTime

from txt_to_speech.txt_to_speech import TxtToSpeech


KEYWORDS = {'marvin', 'marwin', 'sheila'}
KEYWORD_GPT4 = {'turbo'}


class HandleTxtInput:
    def __init__(self):
        self.functions_service = FunctionsService()
        self.gpt = GPTClient([f.description for f in self.functions_service.all_functions.values()])
        self.text_to_speech = TxtToSpeech()

    def has_keywords(self, transcript, keywords):
        words = transcript.replace(',', '').split()
        for keyword in keywords:
            if keyword in words:
                return keyword
        return False

    def remove_keyword(self, txt, keyword):
        words = txt.replace(',', '').split()
        if keyword in words:
            words.remove(keyword)
            txt = ' '.join(words)
        return txt

    def handle_txt_input(self, txt):

        print(txt)
        if txt:
            keyword = self.has_keywords(txt.lower(), KEYWORDS)
            if keyword:
                txt = self.remove_keyword(txt, keyword)
            gpt4_keyword = self.has_keywords(txt, KEYWORD_GPT4)
            if gpt4_keyword:
                txt = self.remove_keyword(txt, gpt4_keyword)
                gpt_version = GPT4
            else:
                gpt_version = GPT3
            if txt:
                print(f"Sending to {gpt_version}")
                MeasureTime.start_measure_function_time('gpt')
                # result_queue = multiprocessing.Queue()
                gpt_result = self.gpt.function_call(txt, None, gpt_version)
                # result_queue.put(result)
                # p = multiprocessing.Process(target=self.gpt.function_call, args=(txt, result_queue, gpt_version))
                # p.start()
                #
                # p.join(30)
                # if p.is_alive():
                #     p.terminate()
                #     p.join()
                time_consumed_gpt = MeasureTime.stop_measure_function_time('gpt')
                print(f'time consumed gpt: {time_consumed_gpt}')
                # if not result_queue.empty():
                #     gpt_result = result_queue.get()
                #     print("Result from the process: ")
                #     print(gpt_result)
                # else:
                #     print("Process did not provide a result within the time limit.")
                #     return
            else:
                print('Empty.')
                return
            if gpt_result:
                try:
                    function_name = gpt_result['name']
                except:
                    function_name = None
                if function_name:
                    if function_name in self.functions_service.all_functions:
                        args = gpt_result['args']
                        self.functions_service.all_functions[function_name](args)
                    else:
                        print(f'Function \'{function_name}\' not found. Available functions:'
                              f' {self.functions_service.all_functions.keys()}')
                else:
                    try:
                        content = gpt_result['content']
                    except:
                        content = "No content."
                    print(f'\n{content}\n')
                    MeasureTime.start_measure_function_time('tts')
                    self.text_to_speech.speak(content)
