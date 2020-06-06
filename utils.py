from functools import wraps
from time import perf_counter


def print_timing(func):
    @wraps(func)
    def wrapper(*arg, **kwargs):
        start = perf_counter()
        result = func(*arg, **kwargs)
        end = perf_counter()
        fs = '{} took {:.3f} milliseconds'
        print(fs.format(func.__name__, (end - start) * 1000))
        return result
    return wrapper
