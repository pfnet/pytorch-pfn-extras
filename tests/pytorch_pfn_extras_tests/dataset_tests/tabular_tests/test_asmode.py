import pytest

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras_tests.dataset_tests.tabular_tests import dummy_dataset  # NOQA


@pytest.mark.parametrize(
    'mode',
    [tuple, dict, None]
)
def test_astuple(mode):
    dataset = dummy_dataset.DummyDataset(mode=mode, convert=True)
    view = dataset.astuple()
    assert isinstance(view, ppe.dataset.TabularDataset)
    assert len(view) == len(dataset)
    assert view.keys == dataset.keys
    assert view.mode == tuple
    assert (
        view.get_examples(None, None) == dataset.get_examples(None, None))
    assert view.convert(view.fetch()) == 'converted'


@pytest.mark.parametrize(
    'mode',
    [tuple, dict, None]
)
def test_asdict(mode):
    dataset = dummy_dataset.DummyDataset(mode=mode, convert=True)
    view = dataset.asdict()
    assert isinstance(view, ppe.dataset.TabularDataset)
    assert len(view) == len(dataset)
    assert view.keys == dataset.keys
    assert view.mode == dict
    assert (
        view.get_examples(None, None) == dataset.get_examples(None, None))
    assert view.convert(view.fetch()) == 'converted'
