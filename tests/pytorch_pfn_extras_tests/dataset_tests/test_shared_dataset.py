import pytorch_pfn_extras as ppe
import torch


class DummySharedDataset(ppe.dataset.SharedDataset):
    def __init__(self):
        self.data = torch.arange(100).reshape(100, 1)
        super().__init__(self.data.shape)

    def __getitem__(self, idx):
        try:
            x = super().__getitem__(idx)
        except ppe.dataset.ItemNotFoundException:
            x = self.data[idx]
            self.cache_item(idx, x)
        return x

    def __len__(self):
        return len(self.data)


def test_empty_shared_dataset():
    dataset = DummySharedDataset()
    for i in range(100):
        assert not dataset.is_cached(i)


def test_shared_dataset():
    dataset = DummySharedDataset()
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=0)
    for _ in dataloader:
        pass
    for i in range(100):
        assert dataset.is_cached(i)
