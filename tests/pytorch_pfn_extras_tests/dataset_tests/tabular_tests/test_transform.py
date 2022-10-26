import itertools

import numpy as np
import pytest

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras_tests.dataset_tests.tabular_tests import dummy_dataset  # NOQA


# filter out invalid combinations of params
def _filter_params(params):
    for param in params:
        if param[1] is None and \
           isinstance(param[3], tuple) and \
           any(1 <= key_index
               for key_index in param[3]):
            continue

        yield param


@pytest.mark.parametrize(
    'in_mode, out_mode, indices, key_indices, with_batch',
    _filter_params(itertools.product(
        [tuple, dict, None],
        [tuple, dict, None],
        [None, [1, 3], slice(None, 2)],
        [None, (0,), (1,), (1, 0)],
        [False, True]))
)
def test_transform(in_mode, out_mode, indices, key_indices, with_batch):
    dataset = dummy_dataset.DummyDataset(
        mode=in_mode, return_array=True, convert=True)

    def transform(*args, **kwargs):
        if in_mode is tuple:
            assert len(args) == 3
            assert len(kwargs) == 0
            a, b, c = args
        elif in_mode is dict:
            assert len(args) == 0
            assert len(kwargs) == 3
            a, b, c = kwargs['a'], kwargs['b'], kwargs['c']
        elif in_mode is None:
            assert len(args) == 1
            assert len(kwargs) == 0
            a, = args
            b, c = a, a

        if with_batch:
            assert isinstance(a, np.ndarray)
            assert isinstance(b, np.ndarray)
            assert isinstance(c, np.ndarray)
        else:
            assert isinstance(a, float)
            assert isinstance(b, float)
            assert isinstance(c, float)

        if out_mode is tuple:
            return a + b, b + c
        elif out_mode is dict:
            return {'alpha': a + b, 'beta': b + c}
        elif out_mode is None:
            return a + b + c

    def transform_alpha(*args, **kwargs):
        if in_mode is tuple:
            assert len(args) == 3
            assert len(kwargs) == 0
            a, b, c = args
        elif in_mode is dict:
            assert len(args) == 0
            assert len(kwargs) == 3
            a, b, c = kwargs['a'], kwargs['b'], kwargs['c']
        elif in_mode is None:
            assert len(args) == 1
            assert len(kwargs) == 0
            a, = args
            b, c = a, a

        if with_batch:
            assert isinstance(a, np.ndarray)
            assert isinstance(b, np.ndarray)
            assert isinstance(c, np.ndarray)
        else:
            assert isinstance(a, float)
            assert isinstance(b, float)
            assert isinstance(c, float)

        if out_mode is tuple:
            return a + b,
        elif out_mode is dict:
            return {'alpha': a + b}
        elif out_mode is None:
            return a + b + c

    def transform_beta(*args, **kwargs):
        if in_mode is tuple:
            assert len(args) == 3
            assert len(kwargs) == 0
            a, b, c = args
        elif in_mode is dict:
            assert len(args) == 0
            assert len(kwargs) == 3
            a, b, c = kwargs['a'], kwargs['b'], kwargs['c']
        elif in_mode is None:
            assert len(args) == 1
            assert len(kwargs) == 0
            a, = args
            b, c = a, a

        if with_batch:
            assert isinstance(a, np.ndarray)
            assert isinstance(b, np.ndarray)
            assert isinstance(c, np.ndarray)
        else:
            assert isinstance(a, float)
            assert isinstance(b, float)
            assert isinstance(c, float)

        if out_mode is tuple:
            return b + c,
        elif out_mode is dict:
            return {'beta': b + c}
        elif out_mode is None:
            return a + b + c

    if in_mode is not None:
        a, b, c = dataset.data
    else:
        a, = dataset.data
        b, c = a, a

    if out_mode is not None:
        if in_mode is not None:
            d_transform = [
                ((('a', 'b', 'c'), ('alpha', 'beta')), transform)]
        else:
            d_transform = [
                ((('a',), ('alpha',)), transform_alpha),
                ((('a',), ('beta',)), transform_beta)]
        if with_batch:
            view = dataset.transform_batch(('alpha', 'beta'), d_transform)
        else:
            view = dataset.transform(('alpha', 'beta'), d_transform)
        data = np.vstack((a + b, b + c))
    else:
        if in_mode is not None:
            d_transform = [
                ((('a', 'b', 'c'), ('alpha',)), transform_alpha)]
        else:
            d_transform = [
                ((('a',), ('alpha',)), transform_alpha)]
        if with_batch:
            view = dataset.transform_batch(('alpha',), d_transform)
        else:
            view = dataset.transform(('alpha',), d_transform)
        data = (a + b + c)[None]

    assert isinstance(view, ppe.dataset.TabularDataset)
    assert len(view) == len(dataset)
    if out_mode is not None:
        assert view.keys == ('alpha', 'beta')
        assert view.mode == out_mode
    else:
        assert view.keys == ('alpha',)
        assert view.mode == out_mode

    output = view.get_examples(indices, key_indices)

    if indices is not None:
        data = data[:, indices]
    if key_indices is not None:
        data = data[list(key_indices)]

    for out, d in itertools.zip_longest(output, data):
        np.testing.assert_equal(out, d)
        if with_batch:
            assert isinstance(out, np.ndarray)
        else:
            assert isinstance(out, list)

    assert view.convert(view.fetch()) == 'converted'


@pytest.mark.parametrize(
    'mode',
    [tuple, dict, None]
)
class TestTransformInvalid:

    def setup_method(self):
        self.count = 0

    def _transform(self, a, b, c):
        self.count += 1
        if self.count % 2 == 0:
            mode = self.mode
        else:
            if self.mode is tuple:
                mode = dict
            elif self.mode is dict:
                mode = None
            elif self.mode is None:
                mode = tuple

        if mode is tuple:
            return a,
        elif mode is dict:
            return {'a': a}
        elif mode is None:
            return a

    def test_transform_inconsistent_mode(self, mode):
        dataset = dummy_dataset.DummyDataset()
        self.mode = mode
        view = dataset.transform(
            ('a',),
            [((('a', 'b', 'c'), ('a',)), self._transform)])
        view.get_examples([0], None)
        with pytest.raises(ValueError):
            view.get_examples([0], None)

    def test_transform_batch_inconsistent_mode(self, mode):
        dataset = dummy_dataset.DummyDataset()
        self.mode = mode
        view = dataset.transform_batch(
            ('a',),
            [((('a', 'b', 'c'), ('a',)), self._transform)])
        view.get_examples(None, None)
        with pytest.raises(ValueError):
            view.get_examples(None, None)

    def test_transform_batch_length_changed(self, mode):
        dataset = dummy_dataset.DummyDataset()
        self.mode = mode

        def transform_batch(a, b, c):
            if self.mode is tuple:
                return a + [0],
            elif self.mode is dict:
                return {'a': a + [0]}
            elif self.mode is None:
                return a + [0]

        view = dataset.transform_batch(
            ('a',),
            [((('a', 'b', 'c'), ('a',)), transform_batch)])
        with pytest.raises(ValueError):
            view.get_examples(None, None)
