from .. import gpu

args = gpu.args

model_load_dir = args.data_path.joinpath("model")
model_load_dir.mkdir(exist_ok=True)

model_save_dir = args.result_output_path.joinpath("model")
model_save_dir.mkdir(exist_ok=False)

log = gpu.log

try:
    import importlib

    config_file_name = args.config.split("-")[0]
    config = importlib.import_module(f".{config_file_name}", package=__package__)
except ModuleNotFoundError:
    log.warning(f"config {args.config} not found, using default config")
    from . import default as config

config_args = config.config_args
hp = config.hp
log.info(f"config: {config_args}")
