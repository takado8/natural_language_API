import os
import time

import openai
import json

from client.gpio_client import APIClient


class GPTClient:
    def __init__(self):
        key = os.environ['GPT_KEY']
        openai.api_key = key

    def function_call(self, message):
        messages = [{"role": "user", "content": message}]
        functions = [
            {
                "name": "switch",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "devices": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["lamp", "desk_lamp", "main_light", "fan"]
                                }
                        }
                    },
                    "required": ["devices"],
                }
            }
        ]
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=functions,
            function_call={"name": "switch"},
            temperature=0,
            timeout=5
        )

        print(completion)
        response_message = completion["choices"][0]["message"]
        #
        # # Step 2: check if GPT wanted to call a function
        function_call = response_message.get("function_call")
        if function_call:
            arguments = function_call.get("arguments")
            if arguments:
                arg_dict = json.loads(arguments)
                print(arg_dict)
                print(type(arg_dict))
                if 'devices' in arg_dict:
                    return arg_dict['devices']

    def send_request(self):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "fill in parameter and generate one line string POST request to call api endpoint: /relay/<device_name>"},
                {"role": "user", "content": "Zgaś lampę"}

            ]
        )

        print(completion)


if __name__ == '__main__':
    gpt = GPTClient()
    device_names = gpt.function_call("apagar la luz principal")
    gpio = APIClient()
    for dev in device_names:
        gpio.switch_device(dev)
        time.sleep(0.5)
