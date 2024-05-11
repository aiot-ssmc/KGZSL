from ..data import args, hp, data_config

config_args = args.config.split("-")

hp.update(lr=1e-4, weight_decay=1e-4, batch_size=6, total_epochs=100, eval_batch_size=1,
          pre_name="STFT", backbone_name="mixed", head_name="ggnn",
          hidden_dim=128, dropout=(0.4, 0.0),
          )

data_config.update(
    sample_method="u",  # "p" or "u" or "t"
    min_duration_seconds=4,
    max_duration_seconds=9,
    eval_duration_seconds=29,
    batch_size_multiplier=3,
)
