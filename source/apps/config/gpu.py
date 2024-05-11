from . import *
import lightning
import dl
import torch

from lightning.fabric import seed_everything

seed_everything(args.random_seed)

tlog = dl.log.Writer(log_dir=args.result_output_path)

if not args.no_release:
    tlog.release_source(utils.file.SOURCE_PATH)

# It may get stuck if TensorBoard is initiated during training.
# tlog.start_tensorboard(port="6906")

if not args.use_cpu:
    import dl.gpu

    if args.cuda_devices is None:
        args.cuda_devices = dl.gpu.auto_choose_cuda(args.cuda_num, waiting=(not args.no_waiting))
    else:
        args.cuda_devices = [int(i) for i in args.cuda_devices.split(',')]
    dl.gpu.output_stat(cuda2use=args.cuda_devices, output_func=log.info)
    dl.gpu.check_stat(cuda2use=args.cuda_devices, output_func=log.warning)

    log.info(f"using cuda:{', '.join(torch.cuda.get_device_name(i) for i in args.cuda_devices)}")
    torch.set_float32_matmul_precision('high')  # highest | high | medium
    fabric = lightning.fabric.Fabric(accelerator="cuda", devices=args.cuda_devices,
                                     precision="32-true")  # precision="bf16-mixed"

else:
    log.info("using cpu")
    fabric = lightning.fabric.Fabric(accelerator="cpu")  # precision="bf16-mixed"

fabric.launch()
# torch.autograd.set_detect_anomaly(True)

utils.output.variable(args, log.info)
