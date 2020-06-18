import itertools

import numpy as np
import pytest

from tests.pytorch_pfn_extras_tests.dataset_tests.tabular_tests import (
    dummy_dataset,
)  # NOQA


@pytest.mark.parametrize(
    "mode, return_array", itertools.product([tuple, dict, None], [True, False])
)
class TestTabularDataset:
    def test_fetch(self, mode, return_array):
        def callback(indices, key_indices):
            assert indices is None
            assert key_indices is None

        dataset = dummy_dataset.DummyDataset(
            mode=mode, return_array=return_array, callback=callback
        )
        output = dataset.fetch()

        if mode is tuple:
            expected = tuple(dataset.data)
        elif mode is dict:
            expected = dict(zip(("a", "b", "c"), dataset.data))
        elif mode is None:
            expected = dataset.data[0]
        np.testing.assert_equal(output, expected)

        if mode is dict:
            output = output.values()
        elif mode is None:
            output = (output,)
        for out in output:
            if return_array:
                assert isinstance(out, np.ndarray)
            else:
                assert isinstance(out, list)

    def test_convert(self, mode, return_array):
        dataset = dummy_dataset.DummyDataset(
            mode=mode, return_array=return_array)
        output = dataset.convert(dataset.fetch())

        if mode is tuple:
            expected = tuple(dataset.data)
        elif mode is dict:
            expected = dict(zip(("a", "b", "c"), dataset.data))
        elif mode is None:
            expected = dataset.data[0]
        np.testing.assert_equal(output, expected)

        if mode is dict:
            output = output.values()
        elif mode is None:
            output = (output,)
        for out in output:
            assert isinstance(out, np.ndarray)

    def test_get_example(self, mode, return_array):
        def callback(indices, key_indices):
            assert indices == [3]
            assert key_indices is None

        dataset = dummy_dataset.DummyDataset(
            mode=mode, return_array=return_array, callback=callback
        )

        if mode is tuple:
            expected = tuple(dataset.data[:, 3])
        elif mode is dict:
            expected = dict(zip(("a", "b", "c"), dataset.data[:, 3]))
        elif mode is None:
            expected = dataset.data[0, 3]

        assert dataset.get_example(3) == expected

    def test_iter(self, mode, return_array):
        dataset = dummy_dataset.DummyDataset(
            mode=mode, return_array=return_array)
        it = iter(dataset)
        for i in range(10):
            if mode is tuple:
                expected = tuple(dataset.data[:, i])
            elif mode is dict:
                expected = dict(zip(("a", "b", "c"), dataset.data[:, i]))
            elif mode is None:
                expected = dataset.data[0, i]

            assert next(it) == expected

        with pytest.raises(StopIteration):
            next(it)
