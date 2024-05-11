from __future__ import annotations

import abc
import io
import pathlib
import pickle
from typing import Optional, Callable, Iterator, Union, Iterable

from PIL import Image
import torch.utils.data
import datasets
import lz4.frame
import numpy
from datasets.utils.typing import PathLike

import utils.module

X_FEATURES_JSON_FILENAME = "x_features.pkl"


class XFeature(abc.ABC):
    decode_when_train = True

    def encode(self, data):
        if data is None:
            return b''
        else:
            return self.encode_data(data)

    def decode(self, x_data):
        if x_data == b'':
            return None
        elif x_data is None:
            return None
        else:
            return self.decode_data(x_data)

    def check(self, data):
        if data is not None:
            assert self.check_equal(self.decode(self.encode(data)), data)

    def encode_data(self, data):
        raise NotImplementedError

    def decode_data(self, x_data):
        raise NotImplementedError

    @staticmethod
    def check_equal(data1, data2) -> bool:
        raise NotImplementedError


class XLabel(XFeature):
    decode_when_train = True

    def __init__(self, label_dtype=numpy.uint16):  # 65535 labels
        from bidict import bidict
        self.label_dtype = label_dtype
        self.label_name = bidict()

    def encode_data(self, name):
        if name not in self.label_name.inverse:
            label = self.label_dtype(len(self.label_name))
            assert label == len(self.label_name), f"Label overflow: {label} != {len(self.label_name)}"
            self.label_name.put(label, name)
        else:
            label = self.label_name.inverse[name]
        return label

    def decode_data(self, label):
        return self.label_name[label]


class XBytes(XFeature):

    def __init__(self, compression_level: int = 5):
        self.compression_level: int = compression_level

    def encode_data(self, data: bytes) -> bytes:
        # x_data = pyarrow.compress(value, 'lz4', asbytes=True)
        x_data = lz4.frame.compress(data, compression_level=self.compression_level)
        return x_data

    def decode_data(self, x_data: bytes) -> bytes:
        data = lz4.frame.decompress(x_data)
        return data

    @staticmethod
    def check_equal(data1, data2) -> bool:
        check_result = (data1 == data2)
        if isinstance(check_result, Iterable):
            return all(check_result)
        else:
            return check_result


class XArray(XBytes):
    def __init__(self, shape: tuple, dtype: str, compression_level: int = 5):
        super().__init__(compression_level=compression_level)
        self.shape = tuple(shape)
        self.dtype = dtype

    def encode_data(self, data: numpy.ndarray) -> bytes:
        compressed_data = super().encode_data(data.tobytes())
        return compressed_data

    def decode_data(self, x_data: bytes) -> numpy.ndarray:
        decompressed_data = super().decode_data(x_data)
        data = numpy.frombuffer(decompressed_data, dtype=self.dtype)
        return data.reshape(self.shape)

    @staticmethod
    def check_equal(data1, data2) -> bool:
        return numpy.array_equal(data1, data2)  # set equal_nan=False to check nan


class XCompress(XFeature):
    def __init__(self, skip_encode=False):
        self.skip_encode = skip_encode
        if self.skip_encode:
            self.encode = self.identity
            self.check = self.check_decode

    @staticmethod
    def identity(data):
        return data if data is not None else b''

    def check_decode(self, x_data):
        if x_data is not None:
            assert isinstance(self.decode(x_data), numpy.ndarray)


if utils.module.installed("soundfile"):
    import soundfile
    from libs.audio import FfmpegAudioCoder


    class XWave(XCompress):
        def __init__(self, compress_fmt='WAV', skip_encode=False, frame_rate=16000, dtype=numpy.float32):
            super().__init__(skip_encode=skip_encode)
            self.compress_fmt = compress_fmt
            if self.compress_fmt == 'MP3':
                self.encode_data = self.get_ffmpeg_encoder()
            elif self.compress_fmt.startswith('MP3'):
                self.encode_data = self.get_ffmpeg_encoder(quality=self.compress_fmt.split('-')[1])
            else:
                assert self.compress_fmt in soundfile.available_formats()
            self.frame_rate = frame_rate
            self.dtype = dtype

        def encode_data(self, data: numpy.ndarray) -> bytes:
            buf = io.BytesIO()
            soundfile.write(buf, data, samplerate=self.frame_rate, format=self.compress_fmt)
            return buf.getvalue()

        @staticmethod
        def get_ffmpeg_encoder(fmt="mp3", quality='high'):
            assert fmt in ('mp3',)
            return FfmpegAudioCoder(quality=quality).encode

        def decode_data(self, x_data: bytes) -> numpy.ndarray:
            buf = io.BytesIO(x_data)
            return soundfile.read(buf, dtype=self.dtype)[0]

        @staticmethod
        def check_equal(data1, data2) -> bool:
            return numpy.abs(data1 - data2).mean() < 0.01


class XImg(XCompress):
    DECODE_ONLY = ['jpeg']

    def __init__(self, compress_fmt: str = 'png', skip_encode=False):  # 'png', 'jpeg'
        super().__init__(skip_encode=skip_encode)
        self.compress_fmt = compress_fmt

    def encode_data(self, data: numpy.ndarray) -> bytes:
        buf = io.BytesIO()
        img = Image.fromarray(data)
        img.save(buf, format=self.compress_fmt)
        return buf.getvalue()

    def decode_data(self, x_data: bytes) -> numpy.ndarray:
        buf = io.BytesIO(x_data)
        img = Image.open(buf, formats=(self.compress_fmt,))
        return numpy.array(img)

    @staticmethod
    def check_equal(data1, data2) -> bool:
        return numpy.array_equal(data1, data2)


def transform(examples, x_features: dict[str, XFeature]):
    for feature, x_obj in x_features.items():
        if x_obj.decode_when_train:
            examples[feature] = [x_obj.decode(feature_data)
                                 for feature_data in examples[feature]]
    return examples


class XDataset(datasets.Dataset):
    def __init__(self, *args, **kwargs):
        super(XDataset, self).__init__(*args, **kwargs)
        self.x_features: dict = {}
        self.have_applied_decoder = False

    def apply_decoder(self):
        if self.have_applied_decoder:
            raise RuntimeError("You have applied decoder!")
        else:
            self.have_applied_decoder = True
            self.set_transform(lambda x: transform(x, self.x_features), columns=list(self.x_features),
                               output_all_columns=True)

    def decode(self, feature_name: str, x_data):
        return self.x_features[feature_name].decode(x_data)

    def de_feature(self, feature_name, data_dict):
        return self.decode(feature_name, data_dict[feature_name])

    def encode(self, feature_name: str, data):
        return self.x_features[feature_name].encode(data)

    def en_feature(self, feature_name, data_dict):
        return self.encode(feature_name, data_dict[feature_name])

    def reset_format(self):
        super(XDataset, self).reset_format()
        self.have_applied_decoder = False

    def filter_by_ids(self, id_name, filtered_ids):
        id_map = {value: index for index, value in enumerate(self[id_name])}
        filtered_index = [id_map[value] for value in filtered_ids]
        return self.select(filtered_index)

    def get_distribution(self, column_name):
        self.unique(column_name)

    @classmethod
    def from_dataset(cls, dataset: datasets.Dataset, x_features: dict) -> XDataset:
        def warp(ds) -> XDataset:
            ds.__class__ = cls
            return ds

        dataset = warp(dataset)
        dataset.x_features = x_features
        dataset.have_applied_decoder = False
        dataset.apply_decoder()
        return dataset

    @classmethod
    def from_generator(
            cls,
            generator: Callable[..., Iterator[dict]],
            x_features=None,
            check: bool = True,
            cache_dir: str = None,
            keep_in_memory: bool = False,
            gen_kwargs: Optional[dict] = None,
            **kwargs
    ):
        x_features = x_features or {}
        features = {}
        for feature_name in x_features:
            if not isinstance(x_features[feature_name], XFeature):
                features[feature_name] = x_features.pop(feature_name)

        if len(features) > 0:
            raise ValueError(f"features {features} is not XFeature")

        def x_generator(**g_kwargs):
            gen = generator(**g_kwargs)
            if check:
                data = next(gen)
                for feature, x_obj in x_features.items():
                    x_obj.check(data[feature])
                    data[feature] = x_obj.encode(data[feature])
                yield data
            for data in gen:
                for feature, x_obj in x_features.items():
                    data[feature] = x_obj.encode(data[feature])
                yield data

        dataset = super().from_generator(
            generator=x_generator,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            gen_kwargs=gen_kwargs,
            **kwargs
        )
        return cls.from_dataset(dataset, x_features)

    def save_to_disk(self, dataset_path: pathlib.Path, **kwargs):
        self.reset_format()
        super(XDataset, self).save_to_disk(dataset_path=str(dataset_path), **kwargs)
        with pathlib.Path(dataset_path).joinpath(X_FEATURES_JSON_FILENAME).open("wb") as x_features_file:
            pickle.dump(self.x_features, x_features_file)
        self.apply_decoder()

    @classmethod
    def load_from_disk(cls, dataset_path: PathLike, keep_in_memory=None, **kwargs) \
            -> XDataset:
        dataset = super().load_from_disk(dataset_path=dataset_path,
                                         keep_in_memory=keep_in_memory, **kwargs)
        with pathlib.Path(dataset_path).joinpath(X_FEATURES_JSON_FILENAME).open("rb") as x_features_file:
            x_features = pickle.load(x_features_file)
        return cls.from_dataset(dataset, x_features)

    def to_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        return TorchDataset(self).get_dataloader(**kwargs)


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: XDataset):
        self.dataset = dataset

    def __getitem__(self, index):
        raise RuntimeError("should never be called")

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, keys):
        return self.dataset[keys]

    @staticmethod
    def collate_fn(items: Union[list, dict]):
        return {name: torch.utils.data.default_collate(item) for name, item in items.items()}

    def get_dataloader(self, **kwargs) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self, **kwargs, collate_fn=self.collate_fn)


class TorchSeqDataset(TorchDataset):
    def __init__(self, dataset: XDataset, *seq_features):
        super().__init__(dataset)
        self.seq_features = seq_features

    def collate_fn(self, items: dict):
        for data_name in self.seq_features:
            data = items[data_name]
            items[data_name] = utils.data.pad_list_array(data)
        return {key: torch.utils.data.default_collate(item) for key, item in items.items()}


class ConcatDataset(torch.utils.data.Dataset):

    def __init__(self, *dataset):
        self.datasets = dataset
        self.ds_lengths = [len(ds) for ds in self.datasets]

    def __len__(self):
        return sum(self.ds_lengths)

    def format_index(self, index):
        if isinstance(index, slice):
            raise NotImplementedError("slicing not supported")
        elif not isinstance(index, int):
            raise TypeError(f"only integers can be used for indexing, not {type(index).__name__}")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range")
        for i, ds_len in enumerate(self.ds_lengths):
            if index < ds_len:
                return i, index
            index -= ds_len

    def __getitem__(self, index):
        if isinstance(index, Iterable):
            raise TypeError("iterable indexing is not supported")
        dsi, index = self.format_index(index)
        return self.datasets[dsi][index]


class XDatasetDict(datasets.DatasetDict):
    @classmethod
    def load_from_disk(cls, dataset_dict_path: PathLike, keep_in_memory=None,
                       storage_options: Optional[dict] = None, **kwargs) -> XDatasetDict:
        from datasets.dataset_dict import fsspec, is_remote_filesystem, extract_path_from_uri, config, json
        fs_token_paths = fsspec.get_fs_token_paths(dataset_dict_path, storage_options=storage_options)
        fs: fsspec.AbstractFileSystem = fs_token_paths[0]

        dataset_dict = XDatasetDict()
        if is_remote_filesystem(fs):
            dest_dataset_dict_path = extract_path_from_uri(dataset_dict_path)
        else:
            fs = fsspec.filesystem("file")
            dest_dataset_dict_path = dataset_dict_path
        dataset_dict_json_path = pathlib.Path(dest_dataset_dict_path, config.DATASETDICT_JSON_FILENAME).as_posix()
        dataset_info_path = pathlib.Path(dest_dataset_dict_path, config.DATASET_INFO_FILENAME).as_posix()
        if fs.isfile(dataset_info_path) and not fs.isfile(dataset_dict_json_path):
            raise FileNotFoundError(
                f"Expected to load a DatasetDict object, but got a Dataset: '{dataset_dict_json_path}'."
            )

        with fs.open(dataset_dict_json_path, "r", encoding="utf-8") as f:
            splits = json.load(f)["splits"]
        for k in splits:
            dataset_dict_split_path = (
                dataset_dict_path.split("://")[0] + "://" + pathlib.Path(dest_dataset_dict_path, k).as_posix()
                if is_remote_filesystem(fs)
                else pathlib.Path(dest_dataset_dict_path, k).as_posix()
            )
            dataset_dict[k] = XDataset.load_from_disk(
                dataset_dict_split_path, keep_in_memory=keep_in_memory, storage_options=storage_options
            )
        return dataset_dict
