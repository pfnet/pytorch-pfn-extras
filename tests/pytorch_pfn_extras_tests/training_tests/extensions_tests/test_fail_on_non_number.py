import numpy
import pytest
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
        return numpy.array([self.values[idx]], numpy.float32), numpy.int64(
            idx % 2
        )


class Model(torch.nn.Module):
    def __init__(self, grad_error):
        super().__init__()
        self.l1 = torch.nn.Linear(1, 3)
        self._grad_error = grad_error

    def forward(self, x):
        y = self.l1(x)
        if self._grad_error:
            return ErroneousFunc.apply(y)
        return y


class ErroneousFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        return i + 1.0

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output + float("nan")


def get_manager_model_optimizer(*, check_grad=True, grad_error=False):
    epochs = 3
    model = Model(grad_error)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizers = {"main": optimizer}
    models = {"main": model}
    manager = training.ExtensionsManager(
        models, optimizers, epochs, iters_per_epoch=4
    )
    manager.extend(FailOnNonNumber(check_grad=check_grad))
    return manager, model, optimizer


def run_train(manager, model, optimizer, *, optimizer_step=True):
    data = Dataset(4)
    data_loader = torch.utils.data.DataLoader(data)
    for _ in range(3):
        with manager.run_iteration():
            for x, y in data_loader:
                x = x.clone().detach().type(torch.float)
                optimizer.zero_grad()
                out = model(x)
                loss = torch.nn.functional.nll_loss(out, y)
                loss.backward()
                if optimizer_step:
                    optimizer.step()


def test_valid():
    manager, model, optimizer = get_manager_model_optimizer()
    run_train(manager, model, optimizer)


def test_nan():
    manager, model, optimizer = get_manager_model_optimizer()
    with torch.no_grad():
        model.l1.weight[1, 0] = float("NaN")
    with pytest.raises(RuntimeError, match="diverge"):
        run_train(manager, model, optimizer)


def test_inf():
    manager, model, optimizer = get_manager_model_optimizer()
    with torch.no_grad():
        model.l1.weight[2, 0] = float("inf")
    with pytest.raises(RuntimeError, match="diverge"):
        run_train(manager, model, optimizer)


def test_check_grad():
    manager, model, optimizer = get_manager_model_optimizer(
        check_grad=True, grad_error=True
    )
    with pytest.raises(RuntimeError, match="diverge"):
        run_train(manager, model, optimizer, optimizer_step=False)


def test_no_check_grad():
    manager, model, optimizer = get_manager_model_optimizer(
        check_grad=False, grad_error=True
    )
    run_train(manager, model, optimizer, optimizer_step=False)
