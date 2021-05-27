import itertools
import operator

import numpy as np
import pytest

import pytorch_pfn_extras as ppe
from tests.pytorch_pfn_extras_tests.dataset_tests.tabular_tests import dummy_dataset  # NOQA

mode_a = [tuple, dict, None]
mode_b = [tuple, dict, None]
return_array = [True, False]
parameter_set = [
    {'indices': None,
     'expected_indices_a': None,
     'expected_indices_b': None},
    {'indices': [3, 1, 4, 12, 14, 13, 7, 5],
     'expected_indices_a': [3, 1, 4, 7, 5],
     'expected_indices_b': [2, 4, 3]},
    {'indices': [3, 1, 4],
     'expected_indices_a': [3, 1, 4]},
    {'indices': slice(13, 6, -2),
     'expected_indices_a': slice(9, 6, -2),
     'expected_indices_b': slice(3, None, -2)},
    {'indices': slice(9, None, -2),
     'expected_indices_a': slice(9, None, -2)},
    {'indices': [1, 2, 1],
     'expected_indices_a': [1, 2, 1]},
    {'indices': []},
]


@pytest.mark.parametrize(
    'mode_a, mode_b, return_array, parameter_set',
    itertools.product(mode_a, mode_b, return_array, parameter_set)
)
def test_concat(mode_a, mode_b, return_array, parameter_set):
    def callback_a(indices, key_indices):
        assert indices == parameter_set['expected_indices_a']
        assert key_indices is None

    dataset_a = dummy_dataset.DummyDataset(
        keys=('a', 'b', 'c') if mode_b else ('a',),
        mode=mode_a,
        return_array=return_array, callback=callback_a,
        convert=True)

    def callback_b(indices, key_indices):
        assert indices == parameter_set['expected_indices_b']
        assert key_indices is None

    dataset_b = dummy_dataset.DummyDataset(
        size=5,
        keys=('a', 'b', 'c') if mode_a else ('a',),
        mode=mode_b,
        return_array=return_array, callback=callback_b)

    view = dataset_a.concat(dataset_b)
    assert isinstance(view, ppe.dataset.TabularDataset)
    assert len(view) == len(dataset_a) + len(dataset_b)
    assert view.keys == dataset_a.keys
    assert view.mode == dataset_a.mode

    output = view.get_examples(parameter_set['indices'], None)

    data = np.hstack((dataset_a.data, dataset_b.data))
    if parameter_set['indices'] is not None:
        data = data[:, parameter_set['indices']]

    for out, d in itertools.zip_longest(output, data):
        np.testing.assert_equal(out, d)
        if return_array and operator.xor(
                ('expected_indices_a' in parameter_set),
                ('expected_indices_b' in parameter_set)):
            assert isinstance(out, np.ndarray)
        else:
            assert isinstance(out, list)

    assert view.convert(output) == 'converted'


def test_concat_key_length():
    dataset_a = dummy_dataset.DummyDataset()
    dataset_b = dummy_dataset.DummyDataset(keys=('a', 'b'))

    with pytest.raises(ValueError):
        dataset_a.concat(dataset_b)


def test_concat_key_order():
    dataset_a = dummy_dataset.DummyDataset()
    dataset_b = dummy_dataset.DummyDataset(keys=('b', 'a', 'c'))

    with pytest.raises(ValueError):
        dataset_a.concat(dataset_b)
