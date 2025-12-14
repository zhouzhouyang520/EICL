import datetime
import time

def convert(n):
    return str(datetime.timedelta(seconds = n)) 

"""A decorator to measure the execution time of decorated functions. If print_log is True, prints function name and execution time."""
def wrapper_calc_time(print_log=True):
    """ 
    :param print_log: Whether to print execution time
    :return: Decorator function
    """

    def wrapper(func):
        def inner_wrapper(*args, **kwargs):
            start_time = time.time()
            func_re = func(*args, **kwargs)
            run_time = time.time() - start_time
            #re_time = f'{func.__name__} execution time: {int(tem_time * 1000)}ms'
            converted_time = convert(run_time)
            if print_log:
                print(f"Function {func.__name__} time:", run_time, converted_time)
            return func_re

        return inner_wrapper

    return wrapper

