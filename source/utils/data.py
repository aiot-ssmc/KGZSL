import math
import random
from typing import Iterable, Union

import numpy
import numpy as np
from scipy.interpolate import interp1d


def normalization(data: numpy.ndarray):
    return (data - data.min()) / (data.max() - data.min())


def smooth_data(data, smooth_level=(2, 5)):
    rounds, size = smooth_level
    d = data
    s_d = d.copy()
    for i in range(rounds):
        for j in range(1, size):
            d += np.concatenate((s_d[j:], np.full(j, s_d[-j])))
        d /= size
        s_d = d.copy()
    return d


def re_sampling(array, sam_num, way='interp', unwrap=False):
    if unwrap:
        array = np.unwrap(array)
    if way == "interp":
        return interp1d(np.arange(len(array)), array)(
            np.arange(0, len(array), len(array) / sam_num))
    elif (way == "avg_ds") | (way == "mid_ds"):
        if way == "mid_ds":
            def way2get_value(x):
                return np.median(x)
        else:
            def way2get_value(x):
                return np.mean(x)
        if sam_num > len(array):
            raise ValueError("sam_num > len(array)")
        avg_num = len(array) // sam_num
        remainder = len(array) % sam_num
        result = np.empty(sam_num)
        for i in range(sam_num):
            if remainder > 0:
                result[i] = way2get_value(array[(i * avg_num):((i + 1) * avg_num + 1)])
                remainder -= 1
            else:
                result[i] = way2get_value(array[i * avg_num:(i + 1) * avg_num])
            i += 1
        return result
    elif way == 'nearest':
        sample_index = np.around(np.linspace(0, len(array) - 1, sam_num, endpoint=True), decimals=0)
        return array[sample_index.astype(int)]


def group_by(array: np.ndarray, indexes):
    if not isinstance(indexes, Iterable):
        indexes = [indexes]
    results = [array]
    for index in indexes:
        results = [r for sr in results for r in group_by_(sr, index)]
    return results


def group_by_(array: np.ndarray, index):
    array = array[(array[:, index]).argsort()]
    return np.split(array, np.cumsum(np.unique(array[:, index], return_counts=True)[1])[:-1])


def isvalid(d):
    return not ((d is None) or (isinstance(d, float) and math.isnan(d)))


def pad_list_array(array_list: list[np.ndarray]):
    max_length = max(len(array) for array in array_list)
    pad_list = [np.pad(array, (0, max_length - len(array))) for array in array_list]
    return pad_list


def random_cut_list_array(array_list: list[np.ndarray], cut_length: int):
    cut_list = []
    for array in array_list:
        if len(array) > cut_length:
            start_index = random.randint(0, len(array) - cut_length)
            cut_list.append(array[start_index:start_index + cut_length])
        else:
            cut_list.append(np.pad(array, (0, cut_length - len(array))))
    return cut_list


class DimReducer:
    X_TYPE = Union[numpy.ndarray, tuple[numpy.ndarray], list[numpy.ndarray]]

    def __init__(self, dim=2, way="PCA", **kwargs):
        self.dim = dim
        if way.upper() == "PCA":
            from sklearn.decomposition import PCA
            self.model = PCA(n_components=self.dim, **kwargs)
        elif way.upper() == "TSNE":
            from sklearn.manifold import TSNE
            self.model = TSNE(n_components=self.dim, **kwargs)
        elif way.upper() == "ICA":
            from sklearn.decomposition import FastICA
            self.model = FastICA(n_components=self.dim, whiten="unit-variance", **kwargs)
        else:
            raise ValueError(f"way {way} is not supported")

    def fit_transform(self, *x: numpy.ndarray):
        """
        Args:
            *x: (n_samples, n_features)
        Return:
            (n_samples, dim)
        """
        y = self.vstack_transform_split(x, self.model.fit_transform)
        return y[0] if len(x) == 1 else y

    def transform(self, *x: numpy.ndarray):
        """
        Args:
            *x: (n_samples, n_features)
        Return:
            (n_samples, dim)
        """
        y = self.vstack_transform_split(x, self.model.transform)
        return y[0] if len(y) == 1 else y

    @staticmethod
    def vstack_transform_split(x: X_TYPE, transform_func) \
            -> Union[numpy.ndarray, list[numpy.ndarray]]:
        if isinstance(x, numpy.ndarray):
            return transform_func(x)
        x_lens = [len(xi) for xi in x]
        stacked_x = numpy.vstack(x)
        y = transform_func(stacked_x)
        y = numpy.split(y, numpy.cumsum(x_lens)[:-1])
        return y

    def reset(self):
        self.model = self.model.__class__(n_components=self.dim)


def reduce_dim(*x: numpy.ndarray, way="ICA", dim=2, **kwargs):
    return DimReducer(dim, way, **kwargs).fit_transform(*x)
