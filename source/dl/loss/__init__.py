__all__ = ['focal', 'multi_label']

import sys
from typing import Union

import torch
from torch import nn, Tensor, optim

from . import *


class Zero(nn.Module):
    @staticmethod
    def forward(*args):
        return torch.zeros_like(args[0]).sum()


class Ignore(nn.Module):
    def __init__(self, loss_func, *ignored_labels):
        super().__init__()
        self.loss_func = loss_func
        self.register_buffer('ignored_labels', torch.tensor(ignored_labels), persistent=False)

    def forward(self, predicts: Tensor, targets: Tensor):
        """
        Args:
            predicts: (batch_size, num_classes)
            targets: (batch_size)
        Returns:
            loss
        """
        mask = ~torch.isin(targets, self.ignored_labels)
        if mask.any():
            return self.loss_func(predicts[mask], targets[mask])
        else:
            return Zero.forward(predicts, targets)


class AutoScheduler(optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer: optim.Optimizer,
                 mode='min',
                 factor: float = 0.1,
                 patience: int = 10,
                 threshold: float = 1e-4,
                 threshold_mode: str = 'rel',
                 cooldown: int = 0,
                 min_lr: Union[list[float], float] = 0,
                 eps: float = 1e-8,
                 verbose: bool = False) -> None:
        super().__init__(optimizer=optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold,
                         threshold_mode=threshold_mode, cooldown=cooldown, min_lr=min_lr, eps=eps, verbose=verbose)

    def step(self, metrics, epoch: int = ...) -> float:
        current_lr = self.optimizers[0].param_groups[0]['lr']
        if current_lr <= 1e-8:
            sys.exit("Learning rate exceeded the minimum value(1e-8).")
        super().step(metrics)
        return current_lr
