import argparse
import os


class Parser(argparse.ArgumentParser):
    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)
        for key, value in args.__dict__.items():
            if isinstance(value, str):
                args.__dict__[key] = value.replace('~', os.environ['HOME'])
        return args
