from ..data import *

config_args = args.config.split("-")

hp.update(lr=2e-4, weight_decay=0e-4, batch_size=8, total_epochs=50, eval_batch_size=24,
          pre_name="STFT", backbone_name="inception", head_name="ggnn",
          hidden_dim=128, dropout=0.5,
          # feature_dim=255,
          )

data_config.update(
    sample_method="u",  # "p" or "u" or "t"
    min_duration_seconds=0.4,
    max_duration_seconds=6,
    eval_duration_seconds=1.0,
    batch_size_multiplier=4,
)
