import time


class MeasureTime:
    functions_execution_time_dict = {}
    starting_time_dict = {}

    @staticmethod
    def start_measure_function_time(name):
        start = time.time()
        MeasureTime.starting_time_dict[name] = start

    @staticmethod
    def stop_measure_function_time(name):
        stop = time.time()
        result = stop - MeasureTime.starting_time_dict[name]
        if name in MeasureTime.functions_execution_time_dict:
            MeasureTime.functions_execution_time_dict[name] += result
        else:
            MeasureTime.functions_execution_time_dict[name] = result

        return result

    @staticmethod
    def print_functions_total_time_consumed():
        for name in MeasureTime.functions_execution_time_dict:
            print(f'{name}: {MeasureTime.functions_execution_time_dict[name]}')

    def __init__(self):
        raise NotImplementedError
