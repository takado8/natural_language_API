from functions.drawer_lock import DrawerLock
from functions.pwm_configurations import PWMDevicesConfigurations
from functions.switch_devices import SwitchDevices


class FunctionsService:
    def __init__(self):
        switch_devices = SwitchDevices()
        drawer_lock = DrawerLock()
        pwm_configurations = PWMDevicesConfigurations()
        self.all_functions = {switch_devices.description['name']: switch_devices,
                              drawer_lock.description['name']: drawer_lock,
                              pwm_configurations.description['name']: pwm_configurations}
