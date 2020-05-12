import pytest
import numpy
import torch

from pytorch_pfn_extras import training

from pytorch_pfn_extras.training.extensions import FailOnNonNumber


class Dataset(torch.utils.data.Dataset):

    def __init__(self, n_data):
        self.values = [i for i in range(n_data)]
        self.n_data = n_data

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return numpy.array(
            [self.values[idx]], numpy.float32), numpy.int64(idx % 2)


def get_manager_model_optimizer():
    epochs = 3
    model = torch.nn.Linear(1, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizers = {'main': optimizer}
    models = {'main': model}
    manager = training.ExtensionsManager(
        models, optimizers, epochs,
        iters_per_epoch=4)
    manager.extend(FailOnNonNumber())
    return manager, model, optimizer


def run_train(manager, model, optimizer):
    data = Dataset(4)
    data_loader = torch.utils.data.DataLoader(data)
    for _ in range(3):
        with manager.run_iteration():
            for x, y in data_loader:
                x = torch.tensor(x, dtype=torch.float)
                optimizer.zero_grad()
                out = model(x)
                loss = torch.nn.functional.nll_loss(out, y)
                loss.backward()
                optimizer.step()


def test_valid():
    manager, model, optimizer = get_manager_model_optimizer()
    run_train(manager, model, optimizer)


def test_nan():
    manager, model, optimizer = get_manager_model_optimizer()
    model.weight[1, 0] = float('NaN')
    with pytest.raises(RuntimeError):
        run_train(manager, model, optimizer)


def test_inf():
    manager, model, optimizer = get_manager_model_optimizer()
    model.weight[2, 0] = float('inf')
    with pytest.raises(RuntimeError):
        run_train(manager, model, optimizer)
