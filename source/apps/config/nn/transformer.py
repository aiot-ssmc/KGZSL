from ..data import args, hp, data_config

config_args = args.config.split("-")

hp.update(lr=2e-5, weight_decay=1e-4, batch_size=6, total_epochs=100, eval_batch_size=1,
          pre_name="STFT", backbone_name="transformer", head_name="simple",
          hidden_dim=128, dropout=0.3,
          )

data_config.update(
    sample_method="u",  # "p" or "u" or "t"
    min_duration_seconds=1,
    max_duration_seconds=5,
    eval_duration_seconds=29,
    batch_size_multiplier=3,
)
