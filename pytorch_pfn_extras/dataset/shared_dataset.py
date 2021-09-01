# mypy: ignore-errors

import ctypes
import multiprocessing

import numpy
import torch


class Cache:
    def is_cached(self, idx):
        raise NotImplementedError

    def add_to_cache(self, idx, x):
        raise NotImplementedError

    def get_value(self, idx):
        raise NotImplementedError


class InfiniteCache(Cache):
    def __init__(self, sm_size):
        super().__init__()
        self.sm_size = sm_size
        total_size = 1
        for x in sm_size:
            total_size *= x
        shared_memory = multiprocessing.Array(ctypes.c_float, total_size)
        storage = numpy.ctypeslib.as_array(shared_memory.get_obj())
        self.storage = storage.reshape(sm_size)
        # This requires a continuous data loader for the cached values not
        # to be lost
        cached_ids = multiprocessing.Array(ctypes.c_bool, sm_size[0])
        self.cached_ids = numpy.ctypeslib.as_array(cached_ids.get_obj())

    def is_cached(self, idx):
        return self.cached_ids[idx] == 1

    def get_value(self, idx):
        x = None
        if self.is_cached(idx):
            x = self.storage[idx]
        return x

    def add_to_cache(self, idx, x):
        self.storage[idx] = x
        self.cached_ids[idx] = 1


class ItemNotFoundException(Exception):
    pass


class SharedDataset(torch.utils.data.Dataset):
    """ Dataset that caches the load samples in shared memory

    Args
    """
    def __init__(self, sm_size, cache_type=InfiniteCache):
        super().__init__()
        self.cache = cache_type(sm_size)

    def __getitem__(self, idx):
        x = self.cache.get_value(idx)
        if x is None:
            raise ItemNotFoundException(
                'Item {} is not in the cache'.format(idx))
        return x

    def is_cached(self, idx):
        return self.cache.is_cached(idx)

    def cache_item(self, idx, x):
        self.cache.add_to_cache(idx, x)
