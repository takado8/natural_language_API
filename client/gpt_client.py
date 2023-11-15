import os
import time

import openai
import json

from client.gpio_client import GPIOClient


class GPTClient:
    def __init__(self, functions_descriptions):
        self.functions_descriptions = functions_descriptions

    def function_call(self, message, result_queue):
        openai.api_key = os.environ['GPT_KEY']
        messages = [{"role": "user", "content": message}]
        functions = self.functions_descriptions

        # print('sending to gpt for completion...')
        completion = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model="gpt-3.5-turbo-1106",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0,
            timeout=15
        )

        print(completion)
        response_message = completion["choices"][0]["message"]
        #
        # # Step 2: check if GPT wanted to call a function
        function_call = response_message.get("function_call")
        if function_call:
            name = function_call.get("name")
            if name:
                arguments = function_call.get("arguments")
                if arguments:
                    arg_dict = json.loads(arguments)
                    print(f'arg_dict: {arg_dict}')
                else:
                    arg_dict = {}

                result_queue.put({'args': arg_dict, 'name': name})
        else:
            content = response_message.get('content')
            result_queue.put({'content': content})

    def send_request(self):
        openai.api_key = os.environ['GPT_KEY']
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[

                {"role": "user", "content": "sup"}
            ]
        )

        print(completion)

        
if __name__ == '__main__':
    gpt = GPTClient(None)
    gpt.send_request()

