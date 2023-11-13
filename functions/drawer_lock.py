import time

from client.tcp_client import TcpClient, ESP_LOCKER_IP
from functions.function_call_abs import FunctionCallABS

tcp = TcpClient(ESP_LOCKER_IP, 8080)


class DrawerLock(FunctionCallABS):
    def __init__(self):
        description = {
            "name": "unlock_drawer",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
        super().__init__(description)

    def __call__(self, *args, **kwargs):
        self.unlock_drawer()

    @staticmethod
    def unlock_drawer():
        tcp.send("open", "GET")
