import itertools

import numpy as np
import pytest

import pytorch_pfn_extras as ppe
from tests.pytorch_pfn_extras_tests.dataset_tests.tabular_tests import dummy_dataset  # NOQA


def _filter_params(params):
    for param in params:
        key_size = 0
        key_size += 3 if param[0] else 1
        key_size += 2 if param[1] else 1

        if param[3] and \
           any(key_size <= key_index for key_index in param[3]):
            continue

        yield param


@pytest.mark.parametrize(
    'mode_a, mode_b, return_array, key_indices',
    _filter_params(itertools.product(
        [tuple, dict, None],
        [tuple, dict, None],
        [True, False],
        [None, (0, 4, 1), (0, 2), (1, 0), ()]))
)
def test_join(mode_a, mode_b, return_array, key_indices):
    if key_indices is None:
        expected_key_indices_a = None
        expected_key_indices_b = None
        return

    key_size_a = 3 if mode_a else 1

    key_indices_a = tuple(
        key_index
        for key_index in key_indices
        if key_index < key_size_a)
    key_indices_b = tuple(
        key_index - key_size_a
        for key_index in key_indices
        if key_size_a <= key_index)

    if key_indices_a:
        expected_key_indices_a = key_indices_a
    if key_indices_b:
        expected_key_indices_b = key_indices_b

    def callback_a(indices, key_indices):
        assert indices is None
        assert key_indices == expected_key_indices_a

    dataset_a = dummy_dataset.DummyDataset(
        mode=mode_a,
        return_array=return_array, callback=callback_a,
        convert=True)

    def callback_b(indices, key_indices):
        assert indices is None
        assert key_indices == expected_key_indices_b

    dataset_b = dummy_dataset. DummyDataset(
        keys=('d', 'e'), mode=mode_b,
        return_array=return_array, callback=callback_b)

    view = dataset_a.join(dataset_b)
    assert isinstance(view, ppe.dataset.TabularDataset)
    assert len(view) == len(dataset_a)
    assert view.keys == dataset_a.keys + dataset_b.keys
    assert view.mode == dataset_a.mode or dataset_b.mode or tuple

    output = view.get_examples(None, key_indices)

    data = np.vstack((dataset_a.data, dataset_b.data))
    if key_indices is not None:
        data = data[list(key_indices)]

    for out, d in itertools.zip_longest(output, data):
        np.testing.assert_equal(out, d)
        if return_array:
            assert isinstance(out, np.ndarray)
        else:
            assert isinstance(out, list)

    assert view.convert(output) == 'converted'


def test_join_length():
    dataset_a = dummy_dataset.DummyDataset()
    dataset_b = dummy_dataset.DummyDataset(size=5, keys=('d', 'e'))

    with pytest.raises(ValueError):
        dataset_a.join(dataset_b)


def test_join_conflict_key():
    dataset_a = dummy_dataset.DummyDataset()
    dataset_b = dummy_dataset.DummyDataset(keys=('a', 'd'))

    with pytest.raises(ValueError):
        dataset_a.join(dataset_b)
