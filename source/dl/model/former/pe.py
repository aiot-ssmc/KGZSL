import abc
import math

import torch
from torch import nn, Tensor

from ... import t2n


class PE(nn.Module):
    def __init__(self, max_seq_len: int, dim: int, add_cls_token: bool = False,
                 dropout: float = 0.1, learnable=True):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dim = dim
        embeddings = self.init_embeddings()
        if learnable:
            self.embeddings = nn.Parameter(embeddings)
        else:
            self.register_buffer('embeddings', embeddings)
        if add_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        else:
            self.cls_token = None
        self.dropout = nn.Dropout(p=dropout)

    def init_embeddings(self) -> Tensor:
        """
        Return shape (1, max_seq_len, dim)
        """
        raise NotImplementedError

    def forward(self, x: Tensor):
        """
        :param x: shape (batch_size, seq_len, dim)
        """
        if self.cls_token is not None:
            cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)
            x = torch.cat([cls_tokens, x], dim=1)
        x += self.embeddings[:, :x.size(1)]
        return self.dropout(x)

    def plot_embeddings(self, ax2p, title='position embedding'):
        """
        Plot the position embedding
        Example:
                >>> pe = Sinusoidal(max_seq_len=5000, dim=1000)
                >>> from matplotlib import pyplot as plt
                >>> fig, ax2plot = plt.gcf(), plt.gca()
                >>> pe.plot_embeddings(ax2plot)
                >>> fig.show()
        """
        im = ax2p.imshow(t2n(self.embeddings.squeeze()), cmap='rainbow', aspect='auto', interpolation='nearest')
        ax2p.figure.colorbar(im, ax=ax2p)
        ax2p.set_title(title)


class Learnable(PE, abc.ABC):
    def __init__(self, max_seq_len: int, dim: int, add_cls_token: bool = False,
                 dropout: float = 0.1, learnable=True):
        assert learnable, f"{self.__class__.__name__} must be learnable."
        super().__init__(max_seq_len, dim, add_cls_token, dropout, learnable=learnable)


class RandN(Learnable):
    def init_embeddings(self):
        return torch.randn(1, self.max_seq_len, self.dim)


class Zero(Learnable):
    def init_embeddings(self):
        return torch.zeros(1, self.max_seq_len, self.dim)


class Sinusoidal(PE):
    def init_embeddings(self):
        position = torch.arange(self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2) * (-math.log(10000.0) / self.dim))
        embeddings = torch.zeros(1, self.max_seq_len, self.dim)
        embeddings[0, :, 0::2] = torch.sin(position * div_term)
        embeddings[0, :, 1::2] = torch.cos(position * div_term)
        return embeddings


class RotaryPE(nn.Module):
    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        position_ids = torch.arange(0, max_seq_len, dtype=torch.float32)
        indices = torch.arange(0, dim // 2, dtype=torch.float32)
        indices = torch.pow(10000.0, -2 * indices / dim)
        sinusoidal_id = torch.einsum("n,d->nd", position_ids, indices)
        sin_pos = sinusoidal_id.sin()[None, :, :]
        cos_pos = sinusoidal_id.cos()[None, :, :]
        self.register_buffer("sin_pos", sin_pos)
        self.register_buffer("cos_pos", cos_pos)

    def apply_rotary(self, x):
        """
        Args:
            x: (batch, seq_len, dim)
        """
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # original rotary PE:
        # return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2, -1)
        # Considering torch.einsum("bmd,bnd->bmn", q, k)， So:
        return torch.cat([x1 * self.cos_pos - x2 * self.sin_pos,
                          x1 * self.sin_pos + x2 * self.cos_pos], dim=-1)

    def forward(self, query, key):
        return self.apply_rotary(query), self.apply_rotary(key)
