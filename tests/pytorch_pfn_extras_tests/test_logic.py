from typing import Any, Mapping
from unittest import mock

import pytest
import pytorch_pfn_extras as ppe
import torch
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from torch.optim import Optimizer


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
        prefix = "train" if self.training else "val"
        loss = F.l1_loss(y, t)
        ppe.reporting.report({prefix + "/loss": loss})
        return loss


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_trainer(device):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    iters_per_epoch = 10
    epochs = 20
    model = MyModel()
    ppe.to(model, device)
    model_with_loss = MyModelWithLossFn(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [
            (
                torch.rand(
                    20,
                ),
                torch.rand(
                    10,
                ),
            )
            for i in range(iters_per_epoch)
        ]
    )
    backward_fn = mock.Mock(return_value=None)

    trainer = ppe.engine.create_trainer(
        model_with_loss,
        optimizer,
        epochs,
        device=device,
        options={"backward_function": backward_fn},
    )
    trainer.run(data)
    assert backward_fn.call_count == epochs * iters_per_epoch


@pytest.mark.parametrize(
    "trigger",
    [
        (1, "epoch"),
        (0.5, "epoch"),
        (10, "iteration"),
        (5, "iteration"),
        (1, "iteration"),
    ],
)
def test_train_step_mode_with_evaluator(trigger):
    iters_per_epoch = 10
    epochs = 20
    model = MyModel()
    ppe.to(model, "cpu")
    model_with_loss = MyModelWithLossFn(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [
            (
                torch.rand(
                    20,
                ),
                torch.rand(
                    10,
                ),
            )
            for i in range(iters_per_epoch)
        ]
    )
    backward_fn = mock.Mock(return_value=None)

    class LogicWithTrainStepCheck(ppe.handler.Logic):
        def train_step(
            self,
            models: Mapping[str, Module],
            optimizers: Mapping[str, Optimizer],
            batch_idx: int,
            batch: Any,
        ) -> Any:
            model = models[self.model_name]
            assert model.training
            return super().train_step(models, optimizers, batch_idx, batch)

    trainer = ppe.engine.create_trainer(
        model_with_loss,
        optimizer,
        epochs,
        logic=LogicWithTrainStepCheck(),
        evaluator=(
            ppe.engine.create_evaluator(
                models=model_with_loss,
                logic=LogicWithTrainStepCheck(),
            ),
            trigger,
        ),
        options={"backward_function": backward_fn},
    )
    trainer.run(data, data)
    assert backward_fn.call_count == epochs * iters_per_epoch
