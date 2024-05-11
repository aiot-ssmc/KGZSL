import torch
from torch import Tensor, nn

import utils
from apps.network import transformer, inception

log = utils.log.get_logger()


class Network(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3, atten_dropout=0.0):
        super().__init__()
        self.inception = inception.Network(hidden_dim, dropout, simplify=True)
        self.transformer = transformer.Network(hidden_dim, atten_dropout, 128, 6, 8)
        self.feature_chunk_size = 50  # 0.5s

    def forward(self, x: Tensor) -> Tensor:
        b_n, s_n, d_n = x.shape
        c_n = s_n // self.feature_chunk_size
        chunked_x = x[:, :self.feature_chunk_size * c_n, :]
        chunked_x = chunked_x.reshape(b_n * c_n, self.feature_chunk_size, d_n)
        chunked_feature = self.inception(chunked_x)
        inc_feature = chunked_feature.reshape(b_n, c_n, d_n)
        trs_feature = self.transformer(inc_feature)
        return trs_feature
