from functions.switch_devices import SwitchDevices


class FunctionsService:
    def __init__(self):
        switch_devices = SwitchDevices()
        self.all_functions = {switch_devices.description['name']: switch_devices}
