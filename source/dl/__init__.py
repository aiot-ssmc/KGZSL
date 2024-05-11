# pip install pynvml tensorboard lz4
__all__ = ['data', 'gpu', 'log', 'model', 'loss', 'x_dataset']

from . import *

import torch

EPS = torch.finfo(torch.float32).eps


def t2n(*tensor: torch.Tensor):
    ndarray = [t.detach().cpu().numpy() for t in tensor]
    if len(ndarray) == 1:
        return ndarray[0]
    else:
        return ndarray


def get_lr(optimizer: torch.optim.Optimizer):
    return optimizer.param_groups[0]['lr']


def get_lrs(optimizer: torch.optim.Optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]


def print_all_parameters(network: torch.nn.Module, network_name='model', out_func=print):
    out_func(f"{network_name}: ")
    network_attrs = network.__dict__.copy()
    # print all modules
    module_dict = network_attrs.pop('_modules', {})
    for name, param in module_dict.items():
        print_all_parameters(param, network_name=f"Module-{name}", out_func=lambda s: out_func(f"\t{s}"))
    # print all parameters
    param_dict = network_attrs.pop('_parameters', {})
    for name, param in param_dict.items():
        out_func(f"\tParameter-{name} ({param.shape}) requires_grad={param.requires_grad}")
    # print all buffers
    buffer_dict = network_attrs.pop('_buffers', {})
    for name, param in buffer_dict.items():
        out_func(f"\tBuffer-{name} ({param.shape}) requires_grad={param.requires_grad}")
    # print other tensors
    for name, param in network_attrs.items():
        if isinstance(param, torch.Tensor):
            out_func(f"\tTensor-{name}: ({param.shape}) requires_grad={param.requires_grad}")


class FreeCacheContext:
    def __enter__(self):
        torch.cuda.empty_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()
