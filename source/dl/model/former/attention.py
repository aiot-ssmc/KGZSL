import math

import torch
from torch import nn, Tensor
from . import mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0, mask_maker: mask.Maker = mask.NoneMaker(), torch_impl=True):
        super().__init__()
        self._dropout_p = dropout
        self.mask_maker = mask_maker
        self.dropout = nn.Dropout(self._dropout_p)
        if torch_impl:
            self.forward = self.torch_forward
        else:
            self.forward = self.custom_forward

    @property
    def dropout_p(self):
        if self.training:
            return self._dropout_p
        else:
            return 0.0

    def custom_forward(self, q: Tensor, k: Tensor, v: Tensor):
        """
        Args:
            q: (batch_size, head_num, seq_len_q, dim)
            k: (batch_size, head_num, seq_len, dim)
            v: (batch_size, head_num, seq_len, dim_v)
        Returns:
            out: (batch_size, head_num, seq_len_q, dim_v)
        """
        attn_weight = (q @ k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        attn_mask = self.mask_maker(attn_weight.size(), attn_weight.device, attn_weight.dtype)
        if attn_mask:
            attn_weight += attn_mask
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.dropout(attn_weight)
        return attn_weight @ v

    def torch_forward(self, q: Tensor, k: Tensor, v: Tensor):
        mask_size = list(q.size())
        mask_size[-1] = k.shape[-2]
        attn_mask = self.mask_maker(torch.Size(mask_size), q.device, q.dtype)
        return nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attn_mask, dropout_p=self.dropout_p, is_causal=False
        )
