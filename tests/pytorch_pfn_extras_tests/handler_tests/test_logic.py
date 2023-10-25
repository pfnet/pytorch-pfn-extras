from copy import deepcopy
from typing import Any, Mapping, Tuple
from unittest import mock

import pytest
import pytorch_pfn_extras as ppe
import torch
from torch import nn
from torch.cuda.amp import GradScaler
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


logic_options_type_map = {
    "backward_outputs": str,
    "grad_scaler": GradScaler,
    "backward_function": None,
    "autocast": bool,
}


@pytest.mark.gpu
@pytest.mark.parametrize(
    "logic_options_name, trainer_options_name",
    [
        ((), ("backward_outputs", "backward_function", "autocast")),
        (("backward_function",), ("backward_function", "autocast")),
        ((), ("backward_outputs", "grad_scaler")),
        (("backward_outputs", "backward_function"), ("backward_outputs",)),
        (("backward_function", "autocast"), ("grad_scaler", "autocast")),
        (
            ("backward_outputs",),
            ("backward_outputs", "grad_scaler", "backward_function"),
        ),
        (
            ("backward_outputs", "grad_scaler", "autocast"),
            ("backward_function",),
        ),
        (("autocast",), ("grad_scaler",)),
        (("grad_scaler", "backward_function"), ()),
        (("backward_function",), ("grad_scaler", "autocast")),
    ],
)
def test_initialize_logic_with_options_with_autocast(
    logic_options_name: Tuple[str, ...], trainer_options_name: Tuple[str, ...]
):
    logic_options = {
        k: mock.MagicMock(spec=logic_options_type_map[k])
        for k in logic_options_name
    }
    trainer_options = {
        k: mock.MagicMock(spec=logic_options_type_map[k])
        for k in trainer_options_name
    }
    expected_options = deepcopy(logic_options)
    expected_options.update(deepcopy(trainer_options))
    expected_logic = ppe.handler.Logic(options=expected_options)

    actual_logic = ppe.handler.Logic(options=logic_options)
    _ = ppe.engine.create_trainer(
        models=mock.MagicMock(spec=nn.Module),
        optimizers=mock.MagicMock(spec=Optimizer),
        max_epochs=1,
        logic=actual_logic,
        options=trainer_options,
    )
    assert actual_logic.backward_outputs == expected_logic.backward_outputs
    assert actual_logic._grad_scaler == expected_logic._grad_scaler
    assert actual_logic._backward_fn == expected_logic._backward_fn
    assert actual_logic._autocast._options == expected_logic._autocast._options


# The autocast option is removed from the test case
# because it will result in a RuntimeError if the gpu is not present.
@pytest.mark.parametrize(
    "logic_options_name, trainer_options_name",
    [
        (
            (),
            (
                "backward_outputs",
                "backward_function",
            ),
        ),
        (("backward_function",), ("backward_function",)),
        ((), ("backward_outputs", "grad_scaler")),
        (("backward_outputs", "backward_function"), ("backward_outputs",)),
        (("backward_function",), ("grad_scaler",)),
        (
            ("backward_outputs",),
            ("backward_outputs", "grad_scaler", "backward_function"),
        ),
        (
            (
                "backward_outputs",
                "grad_scaler",
            ),
            ("backward_function",),
        ),
        ((), ("grad_scaler",)),
        (("grad_scaler", "backward_function"), ()),
        (("backward_function",), ("grad_scaler",)),
    ],
)
def test_initialize_logic_with_options(
    logic_options_name: Tuple[str, ...], trainer_options_name: Tuple[str, ...]
):
    logic_options = {
        k: mock.MagicMock(spec=logic_options_type_map[k])
        for k in logic_options_name
    }
    trainer_options = {
        k: mock.MagicMock(spec=logic_options_type_map[k])
        for k in trainer_options_name
    }
    expected_options = deepcopy(logic_options)
    expected_options.update(deepcopy(trainer_options))
    expected_logic = ppe.handler.Logic(options=expected_options)

    actual_logic = ppe.handler.Logic(options=logic_options)
    _ = ppe.engine.create_trainer(
        models=mock.MagicMock(spec=nn.Module),
        optimizers=mock.MagicMock(spec=Optimizer),
        max_epochs=1,
        logic=actual_logic,
        options=trainer_options,
    )
    assert actual_logic.backward_outputs == expected_logic.backward_outputs
    assert actual_logic._grad_scaler == expected_logic._grad_scaler
    assert actual_logic._backward_fn == expected_logic._backward_fn
    assert actual_logic._autocast._options == expected_logic._autocast._options
