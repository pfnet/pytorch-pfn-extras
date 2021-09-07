# mypy: ignore-errors

from pytorch_pfn_extras.dataset.tabular import tabular_dataset
from pytorch_pfn_extras.dataset.tabular import _utils


class _Slice(tabular_dataset.TabularDataset):

    def __init__(self, dataset, indices, keys):
        if keys is None:
            self._unary = None
        elif isinstance(keys, tuple):
            self._unary = False
        else:
            self._unary = True
            keys = keys,

        self._dataset = dataset
        self._indices = _utils._as_indices(indices, len(dataset))
        self._key_indices = _utils._as_key_indices(keys, dataset.keys)

    def __len__(self):
        if self._indices is None:
            return len(self._dataset)
        elif isinstance(self._indices, slice):
            start, stop, step = self._indices.indices(len(self._dataset))
            return len(range(start, stop, step))
        else:
            return len(self._indices)

    @property
    def keys(self):
        if self._key_indices is None:
            return self._dataset.keys
        else:
            return tuple(self._dataset.keys[key_index]
                         for key_index in self._key_indices)

    @property
    def mode(self):
        if self._unary is None:
            return self._dataset.mode
        elif self._unary:
            return None
        else:
            return self._dataset.mode or tuple

    def get_examples(self, indices, key_indices):
        indices = _utils._merge_indices(
            self._indices, indices, len(self._dataset), len(self))
        key_indices = _utils._merge_key_indices(self._key_indices, key_indices)
        return self._dataset.get_examples(indices, key_indices)

    def convert(self, data):
        return self._dataset.convert(data)


class _SliceHelper(object):

    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, args):
        if isinstance(args, tuple):
            indices, keys = args
        else:
            indices = args
            keys = None

        return _Slice(self._dataset, indices, keys)
