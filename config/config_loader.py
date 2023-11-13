import json


class ConfigLoader:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.config = self.load_config()

    def load_config(self):
        try:
            with open(self.config_file_path, 'r') as file:
                config_data = json.load(file)
            return config_data
        except FileNotFoundError:
            print(f"Error: Config file '{self.config_file_path}' not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Unable to parse JSON in config file '{self.config_file_path}'.")
            return {}

    def get_config(self, key):
        return self.config.get(key, None)
