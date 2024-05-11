import collections
import itertools
from typing import Iterator


class SubIter(object):
    def __init__(self, iterator, num: int):
        self.iterator = iterator
        self.num = num

    def __iter__(self):
        for i, item in zip(range(self.num), self.iterator):
            yield item

    def __len__(self):
        return self.num


def sub(iterator, rate_or_num):
    if rate_or_num < 1.0:
        sub_num = len(iterator) * rate_or_num
    else:
        sub_num = rate_or_num
    return SubIter(iterator, int(sub_num))


def _reiterate(iterator, batch_size: int = 1, random: bool = False, total: int = None):
    if total is not None:
        def get_iter():
            for i, item in enumerate(iterator):
                if i >= total:
                    break
                yield item

        ti = get_iter()
    else:
        ti = iterator
    if random:
        ti = list(ti)
        import random
        random.shuffle(ti)
    yield from batch(ti, batch_size)


def reiterate(dataset, batch_size: int = 1, random: bool = False, total: int = None):
    index_list = range(len(dataset))
    for index_batch in _reiterate(index_list, batch_size, random, total):
        yield [dataset[i] for i in index_batch]


def batch(iterator: Iterator, batch_size: int):
    batch_items = []
    for item in iterator:
        batch_items.append(item)
        if len(batch_items) >= batch_size:
            yield batch_items
            batch_items = []
    if len(batch_items) > 0:
        yield batch_items


def batch_variable(variable_iterator, batch_size: int):
    batch_items = []
    start_index = 0
    while start_index < len(variable_iterator):
        end_index = len(variable_iterator)
        for i in range(start_index, end_index):
            batch_items.append(variable_iterator[i])
            if len(batch_items) >= batch_size:
                yield batch_items
                batch_items = []
        if len(batch_items) > 0:
            yield batch_items
            batch_items = []
        start_index = end_index


def consume(iterator, n=None):
    """Advance the iterator n-steps ahead. If n is none, consume entirely."""
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(itertools.islice(iterator, n, n), None)
