import pytest
import torch

from tests.pytorch_pfn_extras_tests.dataset_tests.tabular_tests import (
    dummy_dataset,
)  # NOQA


@pytest.mark.parametrize(
    'batch_size,mode',
    [(1, dict), (2, dict), (8, dict), (1, tuple), (2, tuple), (8, tuple)],
)
def test_with_dataloader(batch_size, mode):
    size = 10
    keys = ('a', 'b', 'c')
    dataset = dummy_dataset.DummyDataset(size=size, keys=keys, mode=mode)
    expected = torch.tensor(dataset.data).type(torch.float64)
    expected_per_key = [
        [
            expected[i, j * batch_size:(j + 1) * batch_size]
            for j in range((size + batch_size - 1) // batch_size)
        ]
        for i in range(len(keys))
    ]
    print(expected_per_key)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    for i, example in enumerate(dataloader):
        for j, key in enumerate(keys):
            assert torch.allclose(
                expected_per_key[j][i], example[key if mode == dict else j])
