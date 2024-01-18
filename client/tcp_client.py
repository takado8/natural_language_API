import random
import socket
import time

ESP_LOCKER_IP = '192.168.233.114'
ESP_DRYER_IP = '192.168.233.121'
PICO_IP = '192.168.233.113'


class TcpClient:
    POST = 'POST'
    GET = 'GET'
    IP = 'IP'

    def __init__(self, host, port):
        self.host = host
        self.port = port

    def send(self, command, method_type):
        full_command = method_type + ' ' + command
        data = None
        mySocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        mySocket.connect((self.host, self.port))
        mySocket.send(full_command.encode())
        print('{}: sending command: {}'.format(self, full_command))
        if method_type == self.GET:
            data = mySocket.recv(2048).decode()
            print('Respond: ', data)
        mySocket.close()
        return data

    def __str__(self):
        return 'TcpClient(host={}, port={})'.format(self.host, self.port)


if __name__ == '__main__':
    # t = TcpClient(host=ESP_LOCKER_IP, port=8080)
    # t = TcpClient(host=ESP_DRYER_IP, port=8080)
    t = TcpClient(host=PICO_IP, port=8080)
    # time.sleep(1)
    leds = ['r0', 'g0', 'b0']
    configs = [
        {'r0': 100, 'b0': 30, 'g0': 0},
        {'r0': 100, 'b0': 0, 'g0': 20},
        {'r0': 0, 'b0': 100, 'g0': 40},
               ]
    while True:
        for config in configs:
            for color in config:
                t.send(f'{color} {config[color]}', method_type=TcpClient.POST)
            time.sleep(25)
    # t.send("40", led)
    # lock_time = 200
    # t.send(f"lock {lock_time}", "GET")
    # time.sleep(3)
    # normal_mode_time = 60 * 120
    # reversed_mode_time = 60 * 4
    # cycles = 2
    # t.send(f"{normal_mode_time} {reversed_mode_time} {cycles}", "POST")
    # time.sleep(6)
    # t.send("open", "GET")
    # t.send("80", led)
    # t.send(f'{led} 0;', 'POST')

    # time.sleep(3)
    # t.send("all", 'GET')
