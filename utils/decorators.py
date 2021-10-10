from api.utils.logger import MyLogger
from functools import wraps
import time

module_logger = MyLogger(__name__)

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} took {time.time() - t} s.")
        return result

    return wrapper

def timer_log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        module_logger().info(f"Execution took {time.time() - t} s", extra={"func_name_override": func.__name__})
        return result

    return wrapper



def debug(func):
    @wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} returned {result}")
        return result

    return wrapper_debug

def debug_log(func):
    @wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        module_logger().info(f"Calling with arguments: {signature}", extra={"func_name_override": func.__name__})
        result = func(*args, **kwargs)
        module_logger().info(f"Function returned {result}", extra={"func_name_override": func.__name__})
        return result

    return wrapper_debug