import functools
import pathlib

import numpy
import pandas
import torch
from torch import Tensor

import utils.file

if torch.__version__ >= '1.6':
    def torch_stft(input_tensor: Tensor, n_fft: int, hop_length: int = None,
                   win_length: int = None, window: Tensor = None, **kwargs) -> Tensor:
        output = torch.stft(input_tensor, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
                            return_complex=True, **kwargs)
        return torch.stack([output.real, output.imag], dim=-1)


    def torch_istft(input_tensor: Tensor, n_fft: int, hop_length: int = None,
                    win_length: int = None, window: Tensor = None, **kwargs) -> Tensor:
        input_tensor = torch.complex(input_tensor[:, :, :, 0], input_tensor[:, :, :, 1])
        return torch.istft(input_tensor, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window,
                           **kwargs)

else:
    import librosa


    def torch_stft(input_tensor: Tensor, n_fft: int, hop_length: int = None,
                   win_length: int = None, window: Tensor = None, **kwargs) -> Tensor:
        input_data = input_tensor.detach().cpu().numpy()
        output_data = []
        for i in input_data:
            o = librosa.stft(i, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann')
            # output_data.append(o.view(numpy.float32).reshape(o.shape[0], 2, o.shape[1]).transpose(0, 2, 1))
            output_data.append(numpy.stack([o.real, o.imag], axis=-1))
        return torch.from_numpy(numpy.stack(output_data, axis=0))


    def torch_istft(input_tensor: Tensor, n_fft: int, hop_length: int = None,
                    win_length: int = None, window: Tensor = None, **kwargs) -> Tensor:
        input_data = input_tensor.detach().cpu().numpy()
        output_data = []
        for i in input_data:
            i = i[..., 0] + 1j * i[..., 1]
            output_data.append(
                librosa.istft(i, hop_length=hop_length, win_length=win_length, n_fft=n_fft, window='hann'))
        return torch.from_numpy(numpy.stack(output_data, axis=0))


def multi_dim_index(data: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    data.shape = [a1, a2, ..., an, ds, b1, b2, ..., bn]
    index.shape = [a1, a2, ..., an, is]
    result.shape = [a1, a2, ..., an, is, b1, b2, ..., bn]

    Example:
        >>> data_example = torch.arange(40).view(2, 4, 5)
        >>> index_example = torch.tensor([[0, 1, 2],
        >>>                               [1, 2, 3]])
        >>> multi_dim_index(data_example, index_example)
        torch.tensor([
                        [
                            [ 0,  1,  2,  3, 4],
                            [ 5,  6,  7,  8, 9],
                            [10, 11, 12, 13, 14],
                        ],
                        [
                            [25, 26, 27, 28, 29],
                            [30, 31, 32, 33, 34],
                            [35, 36, 37, 38, 39],
                        ]
                    ])
    """
    # get different dimension
    a_shape = index.shape[:-1]
    dim = len(a_shape)
    b_shape = data.shape[dim + 1:]
    a_dims = functools.reduce(lambda x, y: x * y, a_shape)
    b_dims = functools.reduce(lambda x, y: x * y, b_shape)

    data = data.view(a_dims, data.shape[dim], b_dims)
    index = index.view(a_dims, index.shape[dim])

    result = [data[i, index[i, :], :] for i in range(a_dims)]
    result = torch.stack(result, dim=0)
    return result.view(*a_shape, index.shape[1], *b_shape)


def multi_dim_index_write(data: torch.Tensor, index: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """
    data.shape = [a1, a2, ..., an, ds, b1, b2, ..., bn]
    index.shape = [a1, a2, ..., an, is]
    value.shape = [a1, a2, ..., an, is, b1, b2, ..., bn]

    Example:
        >>> data_example = torch.arange(40).view(2, 4, 5)
        >>> index_example = torch.tensor([[0, 1, 2],
        >>>                               [1, 2, 3]])
        >>> value_example =
        >>>     torch.tensor([
        >>>                     [
        >>>                         [ 0,  1,  2,  3, 4],
        >>>                         [ 5,  6,  7,  8, 9],
        >>>                         [10, 11, 12, 13, 14],
        >>>                     ],
        >>>                     [
        >>>                         [25, 26, 27, 28, 29],
        >>>                         [30, 31, 32, 33, 34],
        >>>                         [35, 36, 37, 38, 39],
        >>>                     ]
        >>>                 ])
    """
    # get different dimension
    a_shape = index.shape[:-1]
    dim = len(a_shape)
    b_shape = data.shape[dim + 1:]
    a_dims = functools.reduce(lambda x, y: x * y, a_shape)
    b_dims = functools.reduce(lambda x, y: x * y, b_shape)

    data = data.reshape(a_dims, data.shape[dim], b_dims)
    value = value.view(a_dims, value.shape[dim], b_dims)
    index = index.view(a_dims, index.shape[dim])

    for i in range(a_dims):
        data[i, index[i, :], :] = value[i, :, :]

    return data.view(*a_shape, data.shape[1], *b_shape)


class HParams(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value

    def __repr__(self):
        return 'HParams(' + super(HParams, self).__repr__() + ')'


class InfoFrame:
    # df row number -> label
    # df index -> id
    # df column['name'] -> name
    def __init__(self, data: pandas.DataFrame):
        self.data = data.sort_index()
        self.index = self.data.index
        assert self.index.is_unique

    def filter_by_column(self, column_name: str):
        return self.__class__(self.data[self.data[column_name]])

    def to_tsv(self, path: pathlib.Path):
        self.data.to_csv(path, sep='\t', index=True)

    def to_pkl(self, path: pathlib.Path):
        self.data.to_pickle(path)

    @classmethod
    def load(cls, path: pathlib.Path):
        return cls(utils.file.load_pkl(path))

    @classmethod
    def from_dict(cls, *args, **kwargs):
        return cls(pandas.DataFrame.from_dict(*args, **kwargs))

    @property
    def ids(self):
        return self.index.to_list()

    @property
    def names(self):
        return self.data['name'].to_list()

    def label2id(self, label):
        return self.index.iloc[label]

    def id2label(self, idt, default=-1):
        try:
            return self.index.get_loc(idt)
        except KeyError:
            return default

    @staticmethod
    def index_filter(index: pandas.Index):
        if len(index) == 0:
            return -1
        elif len(index) == 1:
            return index[0]
        else:
            # return index.to_list()
            raise ValueError(f"index {index} has more than one element")

    def column2id(self, column_name: str, value):
        return self.index_filter(self.index[self.data[column_name] == value])

    def column2label(self, column_name: str, value):
        return self.id2label(self.column2id(column_name, value))

    @functools.lru_cache(maxsize=1000)
    def name2id(self, name: str):
        return self.column2id("name", name)

    @functools.lru_cache(maxsize=1000)
    def name2label(self, name: str):
        return self.column2label("name", name)

    def id2column(self, column_name: str, idt):
        return self.data.loc[idt][column_name]

    def label2column(self, column_name: str, label):
        return self.data.iloc[label][column_name]

    def id2name(self, idt):
        return self.id2column("name", idt)

    def label2name(self, label):
        return self.label2column("name", label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
