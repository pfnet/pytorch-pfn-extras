import tempfile

import pytest

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras import engine
from pytorch_pfn_extras import training


@pytest.fixture(scope='function')
def path():
    with tempfile.TemporaryDirectory() as t_path:
        yield t_path


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(20, 15)
        self.l2 = nn.Linear(15, 10)

        with torch.no_grad():
            self.l1.weight.copy_(torch.ones((15, 20)))
            self.l1.bias.copy_(torch.ones((15,)))
            self.l2.weight.copy_(torch.ones((10, 15)))
            self.l2.bias.copy_(torch.ones((10,)))

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = self.l2(y)
        return y


class MyModelWithLossFn(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t):
        y = self.model(x)
        prefix = 'train' if self.training else 'val'
        loss = F.l1_loss(y, t)
        ppe.reporting.report({prefix + '/loss': loss})
        return loss


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_trainer(device, path):
    model = MyModel()
    ppe.to(model, device)
    model_with_loss = MyModelWithLossFn(model)
    ppe.to(model_with_loss, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [(torch.rand(20,), torch.rand(10,)) for i in range(10)])
    extensions = [
        training.extension.ExtensionEntry(
            training.extensions.LogReport(),
            trigger=(10, 'iteration'),
        ),
        training.extensions.ProgressBar(update_interval=2),
    ]

    trainer = engine.create_trainer(
        model_with_loss, optimizer, 20,
        device=device, extensions=extensions,
        out_dir=path,
    )
    trainer.run(data, data)
