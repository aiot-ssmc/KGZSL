# pip install einops
import contextlib
import pathlib
from typing import Union, Iterator, Callable
import torch
from torch import optim
from torch.utils.data import DataLoader

import dl.loss
import utils

MODEL_SUFFIX = ".pt"

log = utils.log.get_logger()


class Module(torch.nn.Module, utils.struct.Object):
    def __init__(self, model_dir: pathlib.Path = None, name: str = None):
        super().__init__()
        self.name = name or utils.module.get_name(self)
        if model_dir:
            self.path = model_dir.joinpath(self.name)
        else:
            self.path = None

    def __post_init__(self):
        self.load_model()

    def load_model(self, model_path=None):
        model_path = model_path if model_path else self.path
        if model_path and model_path.exists():
            for name, module in self._modules.items():
                module_path = model_path / f"{name}{MODEL_SUFFIX}"
                if module_path.exists():
                    log.info(f"Loading model ({name}) from {module_path}")
                    module.load_state_dict(torch.load(module_path, map_location='cpu'))

    def save_model(self, model_dir=None):
        if model_dir:
            path = pathlib.Path(model_dir).joinpath(self.name)
        else:
            path = self.path
        if path is not None:
            path.mkdir(exist_ok=True)
            log.info(f"Saving model to {path}")
            for name, module in self._modules.items():
                if len(module.state_dict()) > 0:
                    torch.save(module.state_dict(), path / f"{name}{MODEL_SUFFIX}")


class Lightning(utils.struct.Object):
    def __init__(self, fabric):
        self.optimizers = None
        self.fabric = fabric
        self.optimizers: list[optim.Optimizer]
        self.schedulers: list[optim.lr_scheduler.ReduceLROnPlateau]

    def __post_init__(self):
        self.optimizers = self.configure_optimizers()
        self.apply2model(self.fabric.setup_module)
        if isinstance(self.optimizers, optim.Optimizer):
            self.optimizer = self.fabric.setup_optimizers(self.optimizers)
            self.optimizers = [self.optimizer, ]
        else:
            self.optimizers = [self.fabric.setup_optimizers(optimizer) for optimizer in self.optimizers]

        self.schedulers = self.configure_schedulers()
        nn_module_list = set(self.model_list) - set(self.auto_model_list)
        if len(nn_module_list) > 0:
            log.info(f"Detected torch.nn.Modules({nn_module_list}) "
                     f"which are not wrapped by dl.model.Module.")

    def configure_optimizers(self):
        raise NotImplementedError

    def configure_schedulers(self):
        raise NotImplementedError

    def zero_grad_all_optimizers(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step_optimizers(self, *args, **kwargs):
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def step_schedulers(self, *args, **kwargs):
        for scheduler in self.schedulers:
            scheduler.step(*args, **kwargs)

    def prepare_dataloaders(self, *dataloaders: DataLoader) -> Union[DataLoader, list[DataLoader]]:
        return self.fabric.setup_dataloaders(*dataloaders)

    def training_batch(self, batch_input, **kwargs) -> torch.Tensor:
        """
        :param batch_input: dict
        :return: loss: torch.Tensor
        """
        raise NotImplementedError

    def training_epoch(self, dataloader, **kwargs):
        """
        :param dataloader: progress_bar(torch.utils.data.DataLoader)
        :return: losses: list[float]
        """
        self.step_schedulers()
        losses = []
        for batch_input in dataloader:
            self.optimizer.zero_grad()
            loss = self.training_batch(batch_input, **kwargs)
            losses.append(loss.item())
            self.fabric.backward(loss)
            self.optimizer.step()
        return losses

    @contextlib.contextmanager
    def evaluating(self):
        self.apply2model(lambda model: model.eval())
        with torch.no_grad():
            yield
        self.apply2model(lambda model: model.train())

    def save_model(self, model_dir=None):
        self.apply2model(lambda model: model.save_model(model_dir), self.auto_model_list)

    @property
    def model_list(self) -> Iterator[str]:
        for model_name in list(self.__dict__.keys()):
            model = self.__dict__[model_name]
            if isinstance(model, torch.nn.Module):
                yield model_name

    @property
    def auto_model_list(self) -> Iterator[str]:
        for model_name in self.model_list:
            model = self[model_name]
            if isinstance(model, Module):
                yield model_name
            elif isinstance(model, torch.nn.Module):  # check if a module wrapped by fabric
                try:
                    model = model.module
                except AttributeError:
                    continue

                if isinstance(model, Module):
                    yield model_name

    def apply2model(self, fn: Callable[[torch.nn.Module], None], model_list=None):
        model_list = model_list or self.model_list
        for model_name in model_list:
            fn_return = fn(self[model_name])
            if fn_return is not None:
                self[model_name] = fn_return

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value
