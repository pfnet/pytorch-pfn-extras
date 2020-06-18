import numpy as np
import pytest

import pytorch_pfn_extras as ppe
from tests.pytorch_pfn_extras_tests.dataset_tests.tabular_tests import dummy_dataset  # NOQA


@pytest.mark.parametrize(
    'mode',
    [tuple, dict, None]
)
def test_with_converter(mode):
    dataset = dummy_dataset.DummyDataset(mode=mode)

    def converter(*args, **kwargs):
        if mode is tuple:
            np.testing.assert_equal(args, tuple(dataset.data))
            assert kwargs == {}
        elif mode is dict:
            assert args == ()
            np.testing.assert_equal(
                kwargs, dict(zip(('a', 'b', 'c'), dataset.data)))
        elif mode is None:
            np.testing.assert_equal(args, tuple(dataset.data))
            assert kwargs == {}

        return 'converted'

    view = dataset.with_converter(converter)
    assert isinstance(view, ppe.dataset.TabularDataset)
    assert len(view) == len(dataset)
    assert view.keys == dataset.keys
    assert view.mode == dataset.mode
    assert (
        view.get_examples(None, None) == dataset.get_examples(None, None))
    assert view.convert(view.fetch()) == 'converted'
