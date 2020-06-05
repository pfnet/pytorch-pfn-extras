import numbers

import numpy as np


def _as_indices(indices, len_):
    if isinstance(indices, slice) or len(indices) == 0:
        return indices

    if all(isinstance(index, (bool, np.bool_)) for index in indices):
        if not len(indices) == len_:
            raise ValueError('The number of booleans is '
                             'different from the length of dataset')
        return [i for i, index in enumerate(indices) if index]
    else:
        checked_indices = []
        for index in indices:
            index = int(index)
            if index < 0:
                index += len_
            if index < 0 or len_ <= index:
                raise IndexError(
                    'index {} is out of bounds for dataset with size {}'
                    .format(index, len_))
            checked_indices.append(index)
        return checked_indices


def _as_key_indices(keys, key_names):
    if keys is None:
        return keys

    key_indices = []
    for key in keys:
        if isinstance(key, numbers.Integral):
            key_index = key
            if key_index < 0:
                key_index += len(key_names)
            if key_index < 0 or len(key_names) <= key_index:
                raise IndexError(
                    'index {} is out of bounds for keys with size {}'.format(
                        key, len(key_names)))
        else:
            try:
                key_index = key_names.index(key)
            except ValueError:
                raise KeyError('{} does not exists'.format(key))
        key_indices.append(key_index)
    return tuple(key_indices)


def _merge_indices(a, b, len_a, len_b):
    if a is None and b is None:
        return None
    elif a is None:
        return b
    elif b is None:
        return a
    elif isinstance(a, slice) and isinstance(b, slice):
        a_start, a_stop, a_step = a.indices(len_a)
        b_start, b_stop, b_step = b.indices(len_b)

        start = a_start + a_step * b_start
        stop = a_start + a_step * b_stop
        step = a_step * b_step

        if start < 0 and step > 0:
            start = None
        if stop < 0 and step < 0:
            stop = None

        return slice(start, stop, step)
    elif isinstance(a, slice):
        a_start, _, a_step = a.indices(len_a)
        return [a_start + a_step * index for index in b]
    elif isinstance(b, slice):
        return a[b]
    else:
        return [a[index] for index in b]


def _merge_key_indices(a, b):
    if a is None and b is None:
        return None
    elif a is None:
        return b
    elif b is None:
        return a
    else:
        return tuple(a[index] for index in b)
