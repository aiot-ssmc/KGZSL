import os
import pathlib
import sys
import time

import utils

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '1, 0'
os.environ["PATH"] += os.pathsep + os.path.dirname(sys.executable)

parser = utils.args.Parser()


def init_args(return_parser=False):
    parser.add_argument('--random_seed', type=int, default=None)

    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--config', type=str, default="default")
    parser.add_argument('--tag', type=str, default=None)

    parser.add_argument('--cpu_num', type=int, default=None)
    parser.add_argument('--enough_memory', action='store_true', default=False)

    parser.add_argument('--log2file', action='store_true', default=False)
    parser.add_argument('--disable_console_log', action='store_true', default=False)
    parser.add_argument('--result_output_path', type=str, default=None)

    parser.add_argument('--use_cpu', action='store_true', default=False)
    parser.add_argument('--cuda_devices', type=str, default=None)
    parser.add_argument('--cuda_num', type=int, default=1)
    parser.add_argument('--no_waiting', action='store_true', default=False)

    parser.add_argument('--no_release', action='store_true', default=False)

    if return_parser:
        return parser
    return parser.parse_args()


args = init_args()

if args.random_seed is None:
    args.random_seed = int(time.time() * 1000) % 1000000

if args.input:
    args.input = pathlib.Path(args.input)
else:
    args.input = utils.file.DATA_PATH.joinpath("input")

config_name = args.config.split("-")[0]

if args.tag is None:
    current_running = pathlib.Path(sys.argv[0])
    if current_running.name.startswith("main"):
        args.tag = args.config
    else:
        args.tag = current_running.parent.name + '.' + current_running.stem + '.' + args.config

if args.data_path:
    args.data_path = pathlib.Path(args.data_path, config_name)
else:
    args.data_path = utils.file.DATA_PATH.joinpath("data", config_name)

args.data_path.mkdir(exist_ok=True)

if args.output:
    args.output = pathlib.Path(args.output, args.tag)
else:
    args.output = utils.file.DATA_PATH.joinpath("output", args.tag)

args.version = time.strftime("%y%m%d%H%M%S", time.localtime())[1:]

if args.result_output_path is None:
    args.result_output_path = args.output.joinpath(args.version)
else:
    args.result_output_path = args.output.joinpath(args.result_output_path)
args.result_output_path.mkdir(parents=True)

utils.log.LOG2Console = ~args.disable_console_log
if args.log2file:
    utils.log.LOG_FILE = args.result_output_path.joinpath('log.txt')

log = utils.log.get_logger("init")

if args.cpu_num is None:
    args.cpu_num = min(os.cpu_count(), 16)


@utils.exit_register
def clear_output_dir():
    dir_size = utils.file.get_size(args.result_output_path)
    log.info(f"{args.result_output_path} size: {dir_size}")
    if dir_size < 10 * 1024:
        log.info("exiting")
        log.info(f'remove {args.result_output_path}')
        utils.file.remove(args.result_output_path)
