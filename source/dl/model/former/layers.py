import einops
from torch import nn

from .attention import ScaledDotProductAttention, mask


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, activation):
        assert activation in ["gelu", "relu"], \
            f"activation({activation}) must be one of 'gelu', 'relu'"

        super().__init__()
        if activation == "gelu":
            activation_layer = nn.GELU()
        else:
            activation_layer = nn.ReLU()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, dim_v: int, head_num: int, bias: bool,
                 dim_head: int, dim_head_v: int, dropout, mask_maker: mask.Maker):
        super().__init__()
        self.head_num = head_num

        self.q_net = nn.Linear(dim, dim_head * head_num, bias=bias)
        self.k_net = nn.Linear(dim, dim_head * head_num, bias=bias)
        self.v_net = nn.Linear(dim_v, dim_head_v * head_num, bias=bias)

        self.mha = ScaledDotProductAttention(dropout=dropout, mask_maker=mask_maker, torch_impl=True)

        self.out_net = nn.Identity() if (head_num == 1 and dim_head_v == dim_v) else nn.Sequential(
            nn.Linear(dim_head_v * head_num, dim_v),
            nn.Dropout(dropout)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_net.weight)
        nn.init.xavier_uniform_(self.k_net.weight)
        nn.init.xavier_uniform_(self.v_net.weight)
        if self.out_net is not nn.Identity:
            nn.init.xavier_uniform_(self.out_net[0].weight)

    def qkv_net(self, q, k, v):
        q, k, v = self.q_net(q), self.k_net(k), self.v_net(v)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.head_num), (q, k, v))
        return q, k, v

    def forward(self, query, key, value):
        """
            Args:
                query: (batch_size, seq_len_q, dim)
                key: (batch_size, seq_len, dim)
                value: (batch_size, seq_len, dim_v)
            Returns:
                out: (batch_size, seq_len_q, dim_v)
        """
        query, key, value = self.qkv_net(query, key, value)
        out = self.mha(query, key, value)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        return self.out_net(out)


class SelfAttention(Attention):

    def __init__(self, dim: int, head_num: int, bias: bool, dim_head: int, dropout: float, mask_maker: mask.Maker):
        super().__init__(dim=dim, dim_v=dim, head_num=head_num, bias=bias,
                         dim_head=dim_head, dim_head_v=dim_head,
                         dropout=dropout, mask_maker=mask_maker)

    def forward(self, x, *args):
        if args:
            raise ValueError("SelfAttention just need one input")
        return super().forward(x, x, x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, head_num, dim_head, dim_ff, dropout=0.0, activation="gelu", bias=True,
                 mask_maker: mask.Maker = mask.NoneMaker(), norm_first=True):
        self.norm_first = norm_first
        super().__init__()
        self.self_attn = SelfAttention(dim=dim, head_num=head_num, dim_head=dim_head, bias=bias,
                                       dropout=dropout, mask_maker=mask_maker)
        self.attn_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim_ff, dropout, activation=activation)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x):
        if self.norm_first:
            x = x + self.self_attn(self.attn_norm(x))
            x = x + self.ff(self.ff_norm(x))
        else:
            x = self.attn_norm(x + self.self_attn(x))
            x = self.ff_norm(x + self.ff(x))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, head_num, dim_head, dim_ff, dropout=0.0, activation="gelu", bias=True,
                 mask_maker: mask.Maker = mask.NoneMaker(), norm="before"):
        assert norm in ["before", "after", None], \
            f"norm({norm}) must be one of 'before', 'after'"
        self.norm_first = (norm != "after")  # default is before

        super().__init__()
        self.self_attn = SelfAttention(dim=dim, head_num=head_num, dim_head=dim_head, bias=bias,
                                       dropout=dropout, mask_maker=mask_maker)
        self.attn_norm = nn.LayerNorm(dim)
        self.cross_attn = Attention(dim=dim, dim_v=dim, head_num=head_num, dim_head=dim_head, bias=bias,
                                    dim_head_v=dim_head, dropout=dropout, mask_maker=mask_maker)
        self.cross_attn_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim_ff, dropout, activation=activation)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x, enc_out):
        if self.norm_first:
            x = x + self.self_attn(self.attn_norm(x))
            x = x + self.cross_attn(self.cross_attn_norm(x), enc_out, enc_out)
            x = x + self.ff(self.ff_norm(x))
        else:
            x = self.attn_norm(x + self.self_attn(x))
            x = self.cross_attn_norm(x + self.cross_attn(x, enc_out, enc_out))
            x = self.ff_norm(x + self.ff(x))
        return x
