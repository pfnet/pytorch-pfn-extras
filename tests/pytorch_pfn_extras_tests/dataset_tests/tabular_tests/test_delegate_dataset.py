import pytest

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.dataset import tabular
from pytorch_pfn_extras_tests.dataset_tests.tabular_tests import dummy_dataset  # NOQA


@pytest.mark.parametrize(
    'mode',
    [tuple, dict, None]
)
def test_delegate_dataset(mode):
    dataset = tabular.DelegateDataset(
        dummy_dataset.DummyDataset(mode=mode))

    assert isinstance(dataset, ppe.dataset.TabularDataset)
    assert len(dataset) == len(dataset.dataset)
    assert dataset.keys == dataset.dataset.keys
    assert dataset.mode == dataset.dataset.mode
    assert (
        dataset.get_example(3) == dataset.dataset.get_example(3))
