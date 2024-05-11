import math
from typing import Union

import torch
from torch import nn, Tensor

from . import layers, pe, mask
from .. import base


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth=6, head_num=16, dim_head=None, dim_ff=None, dropout=0.0, bias=True,
                 activation="gelu", mask_maker: mask.Maker = mask.NoneMaker(), norm_first=True,
                 ):
        super().__init__()
        dim_head = dim_head or dim // int(math.sqrt(head_num))
        dim_ff = dim_ff or dim * 2
        self.transformer = nn.Sequential(*[
            layers.TransformerEncoderLayer(dim=dim, head_num=head_num, dim_head=dim_head, dim_ff=dim_ff,
                                           dropout=dropout, activation=activation, bias=bias,
                                           mask_maker=mask_maker, norm_first=norm_first)
            for _ in range(depth)])

    def forward(self, x: Tensor):
        """
        :param x: (batch_size, seq_len, dim)
        """
        return self.transformer(x)


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth=6, head_num=16, dim_head=None, dim_ff=None, dropout=0.0, bias=True,
                 activation="gelu", mask_maker: mask.Maker = mask.NoneMaker(), norm="before",
                 ):
        super().__init__()
        dim_head = dim_head or dim // int(math.sqrt(head_num))
        dim_ff = dim_ff or dim * 2
        self.transformer = nn.Sequential(*[
            layers.TransformerDecoderLayer(dim=dim, head_num=head_num, dim_head=dim_head, dim_ff=dim_ff,
                                           dropout=dropout, activation=activation, bias=bias,
                                           mask_maker=mask_maker, norm=norm)
            for _ in range(depth)])

    def forward(self, x: Tensor, enc_out: Tensor):
        """
        :param x: (batch_size, seq_len, dim)
        :param enc_out: (batch_size, seq_len, dim)
        """
        for layer in self.transformer:
            x = layer(x, enc_out)
        return x


class Classifier(nn.Module):
    def __init__(self, dim, num_classes=None, depth=6, head_num=16, dropout=0.1, norm_first=True,
                 torch_impl=False, seq_pool: Union[str, nn.Module] = "mean"):
        super().__init__()
        if torch_impl:
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=head_num, batch_first=True, dropout=dropout,
                                                       dim_feedforward=dim * 2, norm_first=norm_first)
            self.encoder = nn.TransformerEncoder(encoder_layer, depth)
        else:
            self.encoder = TransformerEncoder(dim=dim, depth=depth, head_num=head_num, dropout=dropout,
                                              norm_first=norm_first)
        if isinstance(seq_pool, nn.Module):
            self.seq_pool = seq_pool
        elif seq_pool == "mean" or seq_pool == "avg":
            self.seq_pool = nn.Sequential(
                nn.LayerNorm(dim),
                base.Permute(0, 2, 1),
                nn.AdaptiveAvgPool1d(1),
                base.Squeeze(dim=-1)
            )
        elif seq_pool == "max":
            self.seq_pool = nn.Sequential(
                nn.LayerNorm(dim),
                base.Permute(0, 2, 1),
                nn.AdaptiveMaxPool1d(1),
                base.Squeeze(dim=-1)
            )
        else:
            raise ValueError(f"Unknown seq_pool: {seq_pool}")
        if num_classes is None:
            self.mlp_head = nn.Identity()
        else:
            self.mlp_head = nn.Sequential(
                nn.Linear(dim, num_classes)
            )

    def forward(self, x: Tensor) -> torch.Tensor:
        """
        :param x: (batch_size, seq_len, dim)
        """
        output = self.encoder(x)
        pool_output = self.seq_pool(output)
        return self.mlp_head(pool_output)
