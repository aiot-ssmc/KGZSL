# requirements pip install scipy seaborn

__all__ = ['args', 'data', 'file', 'iter', 'log', 'module', 'output', 'plot', 'struct']

import re

from . import *


# requirements: pip install scipy, seaborn

def exit_register(func):
    import atexit
    import signal
    import sys

    def kill_handler():
        func()
        sys.exit(0)

    def wrapper():
        atexit.register(func)
        signal.signal(signal.SIGINT, kill_handler)
        signal.signal(signal.SIGTERM, kill_handler)
        signal.signal(signal.SIGHUP, kill_handler)
        return func()

    return wrapper


def remove_special_char(s, mode='ascii'):
    if mode == 'ascii':
        return s.encode("ascii", "ignore").decode()
    elif mode == 'abc+n':
        return re.sub('[^A-Za-z0-9 ]+', '', s)
    else:
        raise ValueError(f"mode {mode} not supported")


def none_equal(a, b):
    return a != a and b != b or a == b
