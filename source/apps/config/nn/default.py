from ..data import *

config_args = args.config.split("-")

hp.update(lr=1e-3, batch_size=12, total_epochs=100, eval_batch_size=24,
          pre_name="STFT", backbone_name="inception", head_name="simple",
          hidden_dim=128, dropout=0.5,
          )

data_config.update(
    sample_method="u",  # "p" or "u" or "t"
    min_duration_seconds=0.4,
    max_duration_seconds=6,
    eval_duration_seconds=1.0,
    batch_size_multiplier=4,
)
