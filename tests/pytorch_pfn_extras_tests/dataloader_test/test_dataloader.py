import torch

import pytorch_pfn_extras as ppe


class DummyDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        print("init")
        self.data = list(range(10))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # The persistent workers always maintain the original
        # dataset through the dataloader lifetime
        # so the attributes will remain the same as the
        # first time the workers where spawned (dataloader iteration)
        assert self.start == 0
        return self.data[idx]


def test_data_loader_persistent():
    dataset = DummyDataset()
    dataloader = ppe.dataloaders.DataLoader(
        dataset, num_workers=1, persistent_workers=True
    )
    dataset.start = 0
    for i in range(10):
        for _ in dataloader:
            pass
        # Changing the start value here doesn't have any effect in the dataset
        # cached by the workers. since they are not recreated between epochs
        # and can cache values safely
        dataset.start = i
