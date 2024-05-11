from torch import Tensor, nn
from x_transformers import TransformerWrapper, Encoder

import utils

log = utils.log.get_logger()


class Network(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, max_seq_len=1024, depth=12, heads=8, ):
        super().__init__()
        model = TransformerWrapper(
            num_tokens=1,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(
                dim=hidden_dim,
                depth=depth,
                heads=heads,
                layer_dropout=dropout,  # stochastic depth - dropout entire layer
                attn_dropout=dropout,
                ff_dropout=dropout
            )
        )

        model.token_emb = nn.Identity()
        model.to_logits = nn.Identity()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        try:
            feature = self.model(x)
        except UnboundLocalError:
            return self.forward(x)
        return feature.mean(dim=1)
