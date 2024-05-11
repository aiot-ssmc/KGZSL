from typing import Optional

import torch


class Maker:
    def __call__(self, size: torch.Size,
                 device: torch.device,
                 dtype: torch.dtype) -> Optional[torch.Tensor]:
        raise NotImplementedError


class NoneMaker(Maker):
    def __call__(self, size: torch.Size,
                 device: torch.device,
                 dtype: torch.dtype) -> Optional[torch.Tensor]:
        return None


class RealMaker(Maker):

    def get_mask(self, size: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Args:
            size: (batch_size, ..., x, y)
            device: torch.device
            dtype: torch.dtype
        Returns:
            mask: torch.Tensor(x, y)
        """
        raise NotImplementedError

    @staticmethod
    def unsqueeze(mask: torch.Tensor, ref_size: torch.Size) -> torch.Tensor:
        """
        Args:
            mask: torch.Tensor(x, y)
            ref_size: (batch_size, ..., x, y)
        Returns:
            mask: torch.Tensor(1, 1, ..., x, y)
        """
        unsqueeze_dims = (1,) * (len(ref_size) - len(mask.size()))
        return mask.view(*unsqueeze_dims, *mask.size())

    def __call__(self, size: torch.Size,
                 device: torch.device,
                 dtype: torch.dtype) -> torch.Tensor:
        return self.unsqueeze(self.get_mask(size, device, dtype), size)


class FutureInvisible(RealMaker):
    """
    make the future data invisible in attention matrix
    Example:
        >>> mask_maker = FutureInvisible()
        >>> mask_maker(torch.ones(2, 4, 3))
        tensor([[[0.0, -inf, -inf,],
                 [0.0, 0.0, -inf,],
                 [0.0, 0.0, 0.0,],
                 [0.0, 0.0, 0.0,]],)
    """

    def get_mask(self, size: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        r"""
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        use `-inf` instead of 0 because 0 can still yield quite a large softmax weight (since $e^0 = 1$)
        and the mask is applied before softmax.
        """
        mask = (torch.ones(size[-2], size[-1], device=device, dtype=dtype)
                * float('-inf')).triu(diagonal=1)
        return mask


class PreviousInvisible(RealMaker):
    def get_mask(self, size: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError


class FarAwayInvisible(RealMaker):
    def __init__(self, length: int):
        self.length = length

    def get_mask(self, size: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError
