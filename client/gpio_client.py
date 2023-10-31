import requests


class GPIOClient:
    def __init__(self):
        self.BASE_URL = 'http://192.168.233.18:8081'

    def switch_device(self, device_name):
        endpoint = f"{self.BASE_URL}/relay/{device_name}"
        response = requests.post(endpoint)
        return response

    def set_device_state(self, device_name, state):
        endpoint = f"{self.BASE_URL}/relay/{device_name}/{state}"
        response = requests.post(endpoint)
        return response
