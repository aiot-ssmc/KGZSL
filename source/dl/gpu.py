import collections
import os
import time
from dataclasses import dataclass

from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, \
    nvmlDeviceGetMemoryInfo, nvmlShutdown, nvmlDeviceGetName

mem_unit = 1024 * 1024 * 1024  # 1GB

if "CUDA_VISIBLE_DEVICES" in os.environ:
    GPU_ID_A_CUDA = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
else:
    nvmlInit()
    GPU_ID_A_CUDA = list(range(nvmlDeviceGetCount()))
    nvmlShutdown()


@dataclass
class GPUInfo:
    index: int
    cuda_index: int
    cuda_visible: bool
    name: str
    total: float
    used: float
    free: float
    used_percent: float
    busy: bool

    def __str__(self):
        return f"gpu:{self.index}-cuda:{self.cuda_index}({self.name}):\t" \
               f"total: {self.total:4.1f}GB used:{self.used_percent:04.1f}%"


def get_stat():
    nvmlInit()
    gpu_stats = []
    for i in range(nvmlDeviceGetCount()):
        handle = nvmlDeviceGetHandleByIndex(i)
        meminfo = nvmlDeviceGetMemoryInfo(handle)
        name = nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        name = "_".join(name.split(" ")[-2:])
        if i in GPU_ID_A_CUDA:
            cuda_index = GPU_ID_A_CUDA.index(i)
            cuda_visible = True
        else:
            cuda_index = -1
            cuda_visible = False
        gpu_stats.append(
            GPUInfo(
                index=i,
                cuda_index=cuda_index,
                cuda_visible=cuda_visible,
                name=name,
                total=meminfo.total / mem_unit,
                used=meminfo.used / mem_unit,
                free=meminfo.free / mem_unit,
                used_percent=meminfo.used / meminfo.total * 100,
                busy=meminfo.used / meminfo.total > 0.1
            )
        )
    nvmlShutdown()
    return gpu_stats


def output_stat(gpu2use=None, cuda2use=None, output_func=print):
    if cuda2use is not None:
        gpu2use = [GPU_ID_A_CUDA[i] for i in cuda2use]
    if gpu2use is not None:
        output = lambda x: output_func(f"{x}" + ("\t<- using" if x.index in gpu2use else "\t\t"))
    else:
        output = output_func
    collections.deque(map(output, get_stat()), maxlen=0)


def check_stat(gpu2use=None, cuda2use=None, output_func=print):
    stats = get_stat()
    if cuda2use is not None:
        gpu2use = [GPU_ID_A_CUDA[i] for i in cuda2use]
    check_pass = True
    for i in gpu2use:
        if stats[i].busy:
            output_func(f"gpu:{stats[i].index} is busy, "
                        f"total memory:{stats[i].total:.1f}GB, used {stats[i].used_percent:.1f}%")
            check_pass = False
    return check_pass


def auto_choose_cuda(gpu_num=1, reverse=True, ignore_invisible=True, waiting=False):
    while True:
        if ignore_invisible:
            gpu_stats = filter(lambda x: x.cuda_visible, get_stat())
        else:
            gpu_stats = get_stat()
        if waiting:
            gpu_stats = [gpu for gpu in gpu_stats if not gpu.busy]
            if len(gpu_stats) < gpu_num:
                print('.', end='', flush=True)
                time.sleep(5.0)
                continue
        break

    if reverse:
        choices = sorted(gpu_stats, key=lambda gpu: gpu.used_percent, reverse=True)[-gpu_num:]
    else:
        choices = sorted(gpu_stats, key=lambda gpu: gpu.used_percent)[:gpu_num]
    return [gpu.cuda_index for gpu in choices]
