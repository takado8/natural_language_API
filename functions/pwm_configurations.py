import time

from client.tcp_client import TcpClient, PICO_IP
from database.mysql_connector import MySQLConnector
from functions.function_call_abs import FunctionCallABS


class PWMDevicesConfigurations(FunctionCallABS):
    def __init__(self):
        self.tcp = TcpClient(PICO_IP, 8080)

        mysql = MySQLConnector()
        configurations = mysql.get_pwm_configurations()
        self.theme_values = {}
        for c in configurations:
            self.theme_values[c[-1]] = self.config_string_to_dict(c[1])

        description = {
            "name": "set_theme",
            "parameters": {
                "type": "object",
                "properties": {
                    "theme_name": {
                        "type": "string",
                        "enum": list(self.theme_values.keys())
                    }
                },
                "required": ["devices"],
            }
        }
        super().__init__(description)

    def __call__(self, *args, **kwargs):
        print(f'args: {args}')
        self.set_theme(args)

    def set_theme(self, args):
        if args:
            if 'theme_name' in args[0]:
                theme_name = args[0]['theme_name']
                print(f'setting theme: {theme_name} ')
                theme_configs = self.theme_values[theme_name]
                for config in theme_configs:
                    self.tcp.send(f'{config} {theme_configs[config]}', TcpClient.POST)
                return 
        print(f'Error setting theme. args: {args}')

    @staticmethod
    def config_string_to_dict(config_str):
        config_list = config_str.split()
        config_dict = dict(zip(config_list[::2], map(int, config_list[1::2])))
        print(f'config dict: {config_dict}')
        return config_dict
