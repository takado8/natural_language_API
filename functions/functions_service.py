from functions.drawer_lock import DrawerLock
from functions.switch_devices import SwitchDevices


class FunctionsService:
    def __init__(self):
        switch_devices = SwitchDevices()
        drawer_lock = DrawerLock()
        self.all_functions = {switch_devices.description['name']: switch_devices,
                              drawer_lock.description['name']: drawer_lock}
