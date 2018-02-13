import gc
import functools
import sys


def destroy(func):
    """gc装饰器
    :param func:被调用的函数
    :return:None
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print('\n%s' % func.__name__)
        func(*args, **kwargs)
        gc.collect()

    return wrapper
