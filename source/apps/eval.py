import pathlib

import torch

from apps import SAMPLE_RATE
from apps.train import Trainer


def cal_flops(nn: torch.nn.Module):
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(nn, (SAMPLE_RATE,),
                                                 as_strings=True, print_per_layer_stat=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def main():
    trainer = Trainer()
    trainer.model.network.load_model(pathlib.Path("***/apps.network.NN"))
    cal_flops(trainer.model.network.module)
    trainer.eval()


if __name__ == '__main__':
    main()
