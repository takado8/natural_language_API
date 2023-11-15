import time

from client.gpio_client import GPIOClient
from functions.function_call_abs import FunctionCallABS

gpio = GPIOClient()


class SwitchDevices(FunctionCallABS):
    def __init__(self):
        description = {
            "name": "set_states",
            "parameters": {
                "type": "object",
                "properties": {
                    "devices": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["lamp", "desk_lamp", "ceiling_light", "fan"]
                        }
                    },
                    "states": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        }
                    }
                },
                "required": ["devices", "states"],
            }
        }
        super().__init__(description)

    def __call__(self, *args, **kwargs):
        # print(f'args: {args}')
        if 'devices' in args[0] and 'states' in args[0]:
            self.set_states(args[0]['devices'], args[0]['states'])
        else:
            print(f'Invalid arguments: {args}')

    @staticmethod
    def set_states(devices, states):
        print(f'setting {devices} to {states}')
        i = 0
        for device in devices:
            gpio.set_device_state(device, states[i])
            i += 1
