from pytorch_pfn_extras.dataset.tabular import tabular_dataset
from pytorch_pfn_extras.dataset.tabular import _utils


class _TransformBase(tabular_dataset.TabularDataset):
    def __init__(self, dataset, keys, transforms):
        self._dataset = dataset
        key_set = set()

        self._transforms = []
        for s, t in transforms:
            if any(k in key_set for k in s[1]):
                raise ValueError('Transformations must be disjoint')
            key_set.update(s[1])
            ops_idx = _utils._as_key_indices(s[0], self._dataset.keys)
            res_idx = _utils._as_key_indices(s[1], keys)
            self._transforms.append(((ops_idx, res_idx), t))
        if key_set != set(keys):
            raise ValueError(
                'Transformations must produce only all specified keys')

        self._keys = keys

    def __len__(self):
        return len(self._dataset)

    @property
    def keys(self):
        return self._keys

    @property
    def mode(self):
        if not hasattr(self, "_mode"):
            self.get_examples([0], None)
        return self._mode

    def _find_candidate_transforms(self, key_indices):
        # TODO: memoize

        # Assume that all the registered transformations are
        # disjoint on the outputs
        key_indices = list(key_indices)  # sometimes we get ranges
        transforms = []
        operands = set()
        # Look for the transforms that produce the
        # columns specified in the result
        # to avoid calculating uneeded columns
        for s, t in self._transforms:
            ops_idx, res_idx = s
            # We only allow to execute transformations that will generate
            # the exact requested columns
            # We look for transformations that produces the requested keys
            # we allow key_indices select a given key for transformations
            # producting multiple keys since we have ensured all are disjoint
            # contained holds the transf. indexes that belong to key_indices
            # An element that is not required by key_indices,
            # its index is replaced with None.
            contained = [r if r in key_indices else None for r in res_idx]
            if any(r is not None for r in contained):
                # Now look the indices of the keys we need to fetch
                # from the original dataset to apply this transformation
                operands.update(ops_idx)
                transforms.append((ops_idx, t, contained))
        return list(operands), transforms

    def convert(self, data):
        return self._dataset.convert(data)


class _Transform(_TransformBase):

    def get_examples(self, indices, key_indices):
        if key_indices is None:
            key_indices = range(len(self._keys))
        ops_idx, transforms = self._find_candidate_transforms(key_indices)
        in_examples = self._dataset.get_examples(indices, ops_idx)
        out_examples = tuple([] for _ in key_indices)

        for in_example in zip(*in_examples):
            for t_op_idx, transform, t_res_idx in transforms:
                # The size of in_example might not be the same
                # for the transformations.
                # Suppose we have 5 dimensions, a, b, c, d, e
                # Trans 1 uses a, d and Trans 2 uses only c
                # the selection returns a 3 element array of (a,c,d)
                # where trans 1 needs elems 0 and 2 and trans 2 needs 1
                # So we need to select the inputs accordingly
                # Recalculate the ops indexes to reflect this
                inputs = [in_example[ops_idx.index(i)] for i in t_op_idx]
                if self._dataset.mode is tuple:
                    # Should be always be a tuple with the correct value
                    out_example = transform(*inputs)
                elif self._dataset.mode is dict:
                    keys = [self._dataset.keys[i] for i in ops_idx]
                    out_example = transform(
                        **dict(zip(keys, inputs))
                    )
                elif self._dataset.mode is None:
                    out_example = transform(*inputs)
                if isinstance(out_example, tuple):
                    if hasattr(self, "_mode") and self._mode is not tuple:
                        raise ValueError(
                            "transform must not change its return type"
                        )
                    self._mode = tuple
                    for col_index, key_index in enumerate(t_res_idx):
                        if key_index is None:
                            continue
                        # t_res_idx should directly map the output, when
                        # all the outputs are covered this works but when
                        # we are slicing the outputs using key_indices
                        # the result key index needs to be recalculated
                        out_examples[key_indices.index(key_index)].append(
                            out_example[col_index])
                elif isinstance(out_example, dict):
                    if hasattr(self, "_mode") and self._mode is not dict:
                        raise ValueError(
                            "transform must not change its return type"
                        )
                    self._mode = dict
                    for _, key_index in enumerate(t_res_idx):
                        if key_index is None:
                            continue
                        key = self._keys[key_index]
                        out_examples[key_indices.index(key_index)].append(
                            out_example[key]
                        )
                else:
                    if hasattr(self, "_mode") and self._mode is not None:
                        raise ValueError(
                            "transform must not change its return type"
                        )
                    self._mode = None
                    out_example = (out_example,)
                    for col_index, key_index in enumerate(t_res_idx):
                        if key_index is None:
                            continue
                        out_examples[key_indices.index(key_index)].append(
                            out_example[col_index])

        return out_examples

    def convert(self, data):
        return self._dataset.convert(data)


class _TransformBatch(_TransformBase):

    def get_examples(self, indices, key_indices):
        if indices is None:
            len_ = len(self)
        elif isinstance(indices, slice):
            start, stop, step = indices.indices(len(self))
            len_ = len(range(start, stop, step))
        else:
            len_ = len(indices)

        if key_indices is None:
            key_indices = range(len(self._keys))

        ops_idx, transforms = self._find_candidate_transforms(key_indices)
        in_examples = self._dataset.get_examples(indices, ops_idx)
        out_examples = [None for _ in key_indices]
        for t_op_idx, transform, t_res_idx in transforms:
            inputs = [in_examples[ops_idx.index(i)] for i in t_op_idx]
            if self._dataset.mode is tuple:
                out_example = transform(*inputs)
            elif self._dataset.mode is dict:
                keys = [self._dataset.keys[i] for i in ops_idx]
                out_example = transform(
                    **dict(zip(keys, inputs))
                )
            elif self._dataset.mode is None:
                out_example = transform(*inputs)

            if isinstance(out_example, tuple):
                if hasattr(self, "_mode") and self._mode is not tuple:
                    raise ValueError(
                        "transform_batch must not change its return type"
                    )
                self._mode = tuple
                if not all(len(col) == len_ for col in out_example):
                    raise ValueError(
                        "transform_batch must not change the length of data"
                    )
                for col_index, key_index in enumerate(t_res_idx):
                    if key_index is None:
                        continue
                    # t_res_idx should directly map the output, when
                    # all the outputs are covered this works but when
                    # we are slicing the outputs using key_indices
                    # the result key index needs to be recalculated
                    out_examples[key_indices.index(key_index)] = (
                        out_example[col_index])
            elif isinstance(out_example, dict):
                if hasattr(self, "_mode") and self._mode is not dict:
                    raise ValueError(
                        "transform must not change its return type"
                    )
                self._mode = dict
                if not all(len(col) == len_ for col in out_example.values()):
                    raise ValueError(
                        "transform_batch must not change the length of data"
                    )
                for _, key_index in enumerate(t_res_idx):
                    if key_index is None:
                        continue
                    key = self._keys[key_index]
                    out_examples[key_indices.index(key_index)] = (
                        out_example[key])
            else:
                if hasattr(self, "_mode") and self._mode is not None:
                    raise ValueError(
                        "transform must not change its return type"
                    )
                self._mode = None
                out_example = (out_example,)
                if not all(len(col) == len_ for col in out_example):
                    raise ValueError(
                        "transform_batch must not change the length of data"
                    )
                for col_index, key_index in enumerate(t_res_idx):
                    if key_index is None:
                        continue
                    out_examples[key_indices.index(key_index)] = (
                        out_example[col_index])
        return tuple(out_examples)
