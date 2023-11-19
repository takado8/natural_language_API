import os
import time

import openai
import json

from openai import OpenAI
from openai.types.chat import ChatCompletion

from client.gpio_client import GPIOClient


class GPTClient:
    def __init__(self, functions_descriptions):
        self.functions_descriptions = functions_descriptions

    def function_call(self, message, result_queue):
        messages = [{"role": "user", "content": message}]
        functions = self.functions_descriptions

        client = OpenAI(api_key=os.environ['GPT_KEY'])

        completion = client.chat.completions.create(
            # model="gpt-4-1106-preview",
            model="gpt-3.5-turbo-1106",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.6,
            timeout=15
        )

        print(completion)
        response_message = completion.choices[0].message
        function_call = response_message.function_call
        if function_call:
            name = function_call.name
            if name:
                arguments = function_call.arguments
                if arguments:
                    arg_dict = json.loads(arguments)
                    print(f'arg_dict: {arg_dict}')
                else:
                    arg_dict = {}

                result_queue.put({'args': arg_dict, 'name': name})
        else:
            content = response_message.content
            result_queue.put({'content': content})
