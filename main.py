from client.gpio_client import GPIOClient
from client.gpt_client import GPTClient
from speech_to_txt.speech_to_txt import SpeechToTxt
import multiprocessing
import time

KEYWORD = 'heniu'

if __name__ == '__main__':
    sp = SpeechToTxt()
    gpt = GPTClient()
    gpio = GPIOClient()

    while True:
        transcript = sp.listen()
        print(transcript)
        if transcript and KEYWORD in transcript.lower():
            print("Sending to gpt...")
            result_queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=gpt.function_call, args=(transcript, result_queue))
            p.start()
            p.join(20)

            if p.is_alive():
                print("running... let's kill it...")
                # Terminate - may not work if process is stuck for good
                p.terminate()
                # OR Kill - will work for sure, no chance for process to finish nicely however
                # p.kill()
                p.join()
            if not result_queue.empty():
                gpt_result = result_queue.get()
                print("Result from the process: ", gpt_result)
            else:
                print("Process did not provide a result within the time limit.")
                continue

            if gpt_result and isinstance(gpt_result, list):
                for device in gpt_result:
                    gpio.switch_device(device)
                    time.sleep(0.35)
