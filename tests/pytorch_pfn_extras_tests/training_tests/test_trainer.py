import os
import tempfile
import typing
from unittest import mock

import pytest
import pytorch_pfn_extras as ppe
import torch
from pytorch_pfn_extras import engine, training
from pytorch_pfn_extras.training import triggers
from torch import nn
from torch.nn import functional as F


@pytest.fixture(scope="function")
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
        prefix = "train" if self.training else "val"
        loss = F.l1_loss(y, t)
        ppe.reporting.report({prefix + "/loss": loss})
        return loss


def _make_extensions():
    return [
        training.extensions.LogReport(trigger=(10, "iteration")),
        training.extensions.ProgressBar(update_interval=2),
        training.extensions.PrintReport(
            [
                "epoch",
                "iteration",
                "train/loss",
                "val/loss",
                "val/accuracy",
                "elapsed_time",
                "time",
            ]
        ),
    ]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_trainer(device, path):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
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
            for i in range(10)
        ]
    )
    extensions = _make_extensions()

    trainer = engine.create_trainer(
        model_with_loss,
        optimizer,
        20,
        device=device,
        extensions=extensions,
        out_dir=path,
    )
    trainer.run(data)


def test_trainer_no_to(path):
    model = MyModel()
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
            for i in range(10)
        ]
    )
    extensions = _make_extensions()

    trainer = engine.create_trainer(
        model_with_loss,
        optimizer,
        20,
        device="cpu",
        extensions=extensions,
        out_dir=path,
    )
    with pytest.raises(RuntimeError, match="ppe.to"):
        trainer.run(data, data)


def test_trainer_invalid_options(path):
    device = "cpu"
    model = MyModel()
    ppe.to(model, device)
    model_with_loss = MyModelWithLossFn(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    extensions = _make_extensions()
    options = {"UNKNOWN_OPTIONS": True}
    with pytest.raises(ValueError, match="UNKNOWN_OPTIONS"):
        engine.create_trainer(
            model_with_loss,
            optimizer,
            20,
            device=device,
            extensions=extensions,
            out_dir=path,
            options=options,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("progress_bar", [True, False])
def test_train_with_evaluator(device, progress_bar, path):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
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
            for i in range(10)
        ]
    )
    extensions = _make_extensions()

    evaluator = engine.create_evaluator(
        model_with_loss, device=device, progress_bar=progress_bar
    )

    trainer = engine.create_trainer(
        model_with_loss,
        optimizer,
        20,
        device=device,
        evaluator=evaluator,
        extensions=extensions,
        out_dir=path,
    )
    mpath = "pytorch_pfn_extras.training._evaluator.Evaluator.run"
    with mock.patch(mpath) as patched:
        trainer.run(data, data)
        assert patched.call_count == 20


@pytest.mark.parametrize(
    "evaluator_trigger",
    [(20, (1, "epoch")), (40, (5, "iteration"))],
)
def test_evaluator_trigger(evaluator_trigger, path):
    device = "cpu"
    progress_bar = False
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
            for i in range(10)
        ]
    )
    extensions = _make_extensions()

    evaluator = engine.create_evaluator(
        model_with_loss, device=device, progress_bar=progress_bar
    )

    trainer = engine.create_trainer(
        model_with_loss,
        optimizer,
        20,
        device=device,
        evaluator=(evaluator, evaluator_trigger[1]),
        extensions=extensions,
        out_dir=path,
    )
    path = "pytorch_pfn_extras.training._evaluator.Evaluator.run"
    with mock.patch(path) as patched:
        trainer.run(data, data)
        assert patched.call_count == evaluator_trigger[0]


def test_evaluator_dict(path):
    device = "cpu"
    progress_bar = False
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
            for i in range(10)
        ]
    )
    extensions = _make_extensions()

    evaluator1 = engine.create_evaluator(
        model_with_loss, device=device, progress_bar=progress_bar
    )
    evaluator2 = engine.create_evaluator(
        model, device=device, progress_bar=progress_bar
    )

    trainer = engine.create_trainer(
        model_with_loss,
        optimizer,
        20,
        device=device,
        evaluator={
            "1": evaluator1,  # called 20 times.
            "2": (evaluator2, (5, "iteration")),  # called 40 times.
        },
        extensions=extensions,
        out_dir=path,
    )
    path = "pytorch_pfn_extras.training._evaluator.Evaluator.run"
    with mock.patch(path) as patched:
        trainer.run(data, data)
        assert patched.call_count == 20 + 40


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_train_result_equal(device, path):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    train_data = torch.utils.data.DataLoader(
        [
            (
                torch.rand(
                    20,
                ),
                torch.rand(
                    10,
                ),
            )
            for i in range(10)
        ]
    )
    data = torch.utils.data.DataLoader(
        [
            (
                torch.rand(
                    20,
                ),
            )
            for i in range(30)
        ]
    )

    def get_result_from_trainer():
        model = MyModel()
        ppe.to(model, device)
        model_with_loss = MyModelWithLossFn(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        extensions = _make_extensions()

        trainer = engine.create_trainer(
            model_with_loss,
            optimizer,
            20,
            device=device,
            extensions=extensions,
            out_dir=path,
        )
        trainer.run(train_data)

        model.eval()
        with torch.no_grad():
            return [model(x.to(device)) for x, in data]

    def get_result_from_training_loop():
        model = MyModel()
        ppe.to(model, device)
        model_with_loss = MyModelWithLossFn(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        model_with_loss.train()
        for _ in range(20):
            for x, t in train_data:
                optimizer.zero_grad()
                loss = model_with_loss(x.to(device), t.to(device))
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            return [model(x.to(device)) for x, in data]

    result_from_trainer = get_result_from_trainer()
    result_from_training_loop = get_result_from_training_loop()
    for a, e in zip(result_from_trainer, result_from_training_loop):
        assert not a.requires_grad
        assert torch.equal(a, e)


def _compare_states(s1, s2, strict=False):
    def allclose(a, b):
        if strict:
            return (a == b).all()
        else:
            return torch.allclose(a, b)

    if isinstance(s1, dict):
        keys = s1.keys()
        if set(keys) != set(s2.keys()):
            return False
    elif isinstance(s1, (list, tuple)):
        keys = range(len(s1))
        if len(s1) != len(s2):
            return False

    all_equal = True
    for k in keys:
        if isinstance(s1[k], dict):
            if not isinstance(s2[k], dict):
                return False
            all_equal = all_equal and _compare_states(s1[k], s2[k])
        elif isinstance(s1[k], (list, tuple)):
            if not isinstance(s2[k], (list, tuple)):
                return False
            all_equal = all_equal and _compare_states(s1[k], s2[k])
        elif isinstance(s1[k], torch.Tensor):
            all_equal = all_equal and allclose(s1[k], s2[k])
        else:
            all_equal = all_equal and s1[k] == s2[k]
        if not all_equal:
            return all_equal
    return all_equal


class TestTrainerState:
    def _get_trainer(
        self,
        epochs,
        out_dir,
        extensions=None,
        options=None,
        device="cpu",
        grad_scaler=None,
    ):
        model = MyModel()
        ppe.to(model, device)
        model_with_loss = MyModelWithLossFn(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        extensions = extensions or _make_extensions()
        trainer = engine.create_trainer(
            model_with_loss,
            optimizer,
            epochs,
            grad_scalers=grad_scaler,
            device=device,
            extensions=extensions,
            out_dir=out_dir,
            options=options,
        )
        return trainer

    def test_trainer_state(self, path):
        torch.manual_seed(0)
        trainer = self._get_trainer(20, path)
        data = torch.utils.data.DataLoader(
            [
                (
                    torch.ones(
                        20,
                    ),
                    torch.ones(
                        10,
                    ),
                )
                for i in range(10)
            ]
        )
        trainer.run(data)
        # State to be compared to
        state = trainer.state_dict()
        torch.manual_seed(0)
        new_trainer = self._get_trainer(10, path)
        new_trainer.run(data)
        assert not _compare_states(state, new_trainer.state_dict())
        new_trainer = self._get_trainer(20, path)
        new_trainer.load_state_dict(trainer.state_dict())
        new_trainer.run(data)
        assert _compare_states(state, new_trainer.state_dict())

    def test_trainer_autoload(self, path):
        trainer = self._get_trainer(20, path)
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
                for i in range(10)
            ]
        )
        trainer.extend(ppe.training.extensions.snapshot())
        trainer.run(data)

        new_trainer = self._get_trainer(20, path)
        new_trainer.extend(ppe.training.extensions.snapshot(autoload=True))
        # This forces engine initialization
        new_trainer._setup_manager(len(data))
        assert new_trainer.epoch == 20
        assert _compare_states(trainer.state_dict(), new_trainer.state_dict())

    def test_trainer_autoload_training_results_consistency(self, path):
        snapshot_epoch = 10
        training_epoch = 20
        trainer = self._get_trainer(training_epoch, path)
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
                for i in range(10)
            ]
        )
        trainer.extend(
            ppe.training.extensions.snapshot(),
            trigger=triggers.ManualScheduleTrigger([snapshot_epoch], "epoch"),
        )
        trainer.run(data)
        new_trainer = self._get_trainer(training_epoch, path)
        new_trainer.extend(ppe.training.extensions.snapshot(autoload=True))
        new_trainer._setup_manager(len(data))
        assert new_trainer.epoch == snapshot_epoch
        new_trainer.run(data)
        assert new_trainer.epoch == training_epoch
        print(trainer.state_dict().keys())
        trainer_state_dict = trainer.state_dict()
        new_trainer_state_dict = new_trainer.state_dict()
        assert _compare_states(
            trainer_state_dict["models"],
            new_trainer_state_dict["models"],
            strict=True,
        )

    @pytest.mark.gpu
    def test_trainer_autoload_training_results_consistency_with_gpu(self, path):
        if not torch.cuda.is_available():
            pytest.skip()
        snapshot_epoch = 10
        training_epoch = 20
        trainer = self._get_trainer(training_epoch, path, device="cuda")
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
                for i in range(10)
            ]
        )
        trainer.extend(
            ppe.training.extensions.snapshot(),
            trigger=triggers.ManualScheduleTrigger([snapshot_epoch], "epoch"),
        )
        trainer.run(data)
        new_trainer = self._get_trainer(training_epoch, path, device="cuda")
        new_trainer.extend(ppe.training.extensions.snapshot(autoload=True))
        new_trainer._setup_manager(len(data))
        assert new_trainer.epoch == snapshot_epoch
        new_trainer.run(data)
        assert new_trainer.epoch == training_epoch
        print(trainer.state_dict().keys())
        trainer_state_dict = trainer.state_dict()
        new_trainer_state_dict = new_trainer.state_dict()
        assert _compare_states(
            trainer_state_dict["models"],
            new_trainer_state_dict["models"],
            strict=True,
        )

    @pytest.mark.gpu
    def test_trainer_autoload_training_results_no_consistency_with_gradscaler(
        self, path
    ):
        if not torch.cuda.is_available():
            pytest.skip()
        snapshot_epoch = 10
        training_epoch = 20
        grad_scaler = torch.cuda.amp.grad_scaler.GradScaler(
            init_scale=2**48, growth_interval=2
        )
        with pytest.warns(DeprecationWarning):
            trainer = self._get_trainer(
                training_epoch,
                path,
                options={
                    "grad_scaler": grad_scaler,
                },
                device="cuda",
            )
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
                for i in range(10)
            ]
        )
        trainer.extend(
            ppe.training.extensions.snapshot(),
            trigger=triggers.ManualScheduleTrigger([snapshot_epoch], "epoch"),
        )

        trainer.run(data)

        new_grad_scaler = torch.cuda.amp.grad_scaler.GradScaler(
            init_scale=2**48, growth_interval=2
        )
        with pytest.warns(DeprecationWarning):
            new_trainer = self._get_trainer(
                training_epoch,
                path,
                options={
                    "grad_scaler": new_grad_scaler,
                },
                device="cuda",
            )
        new_trainer.extend(ppe.training.extensions.snapshot(autoload=True))
        new_trainer._setup_manager(len(data))
        assert new_trainer.epoch == snapshot_epoch

        new_trainer.run(data)
        assert new_trainer.epoch == training_epoch

        trainer_state_dict = trainer.state_dict()
        new_trainer_state_dict = new_trainer.state_dict()

        # grad_scaler does not store state_dict, so the behavior may change.
        assert not _compare_states(
            trainer_state_dict["models"],
            new_trainer_state_dict["models"],
            strict=True,
        )

    @pytest.mark.gpu
    def test_trainer_autoload_training_results_consistency_with_gradscaler(
        self, path
    ):
        if not torch.cuda.is_available():
            pytest.skip()
        snapshot_epoch = 10
        training_epoch = 20
        grad_scaler = torch.cuda.amp.grad_scaler.GradScaler(
            init_scale=2**48, growth_interval=2
        )
        trainer = self._get_trainer(
            training_epoch,
            path,
            device="cuda",
            grad_scaler=grad_scaler,
        )
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
                for i in range(10)
            ]
        )
        trainer.extend(
            ppe.training.extensions.snapshot(),
            trigger=triggers.ManualScheduleTrigger([snapshot_epoch], "epoch"),
        )

        trainer.run(data)

        new_grad_scaler = torch.cuda.amp.grad_scaler.GradScaler(
            init_scale=2**48, growth_interval=2
        )
        new_trainer = self._get_trainer(
            training_epoch,
            path,
            device="cuda",
            grad_scaler=new_grad_scaler,
        )
        new_trainer.extend(ppe.training.extensions.snapshot(autoload=True))
        new_trainer._setup_manager(len(data))
        assert new_trainer.epoch == snapshot_epoch

        new_trainer.run(data)
        assert new_trainer.epoch == training_epoch

        trainer_state_dict = trainer.state_dict()
        new_trainer_state_dict = new_trainer.state_dict()

        assert _compare_states(
            trainer_state_dict["models"],
            new_trainer_state_dict["models"],
            strict=True,
        )


class MyModelWithLossDictOutput(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t):
        y = self.model(x)
        prefix = "train" if self.training else "val"
        loss = F.l1_loss(y, t)
        ppe.reporting.report({prefix + "/loss": loss})
        return {"y": y, "loss": loss}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("progress_bar", [True, False])
def test_trainer_dict_input(device, progress_bar, path):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = MyModel()
    ppe.to(model, device)
    model_with_loss = MyModelWithLossDictOutput(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [
            {
                "x": torch.rand(
                    20,
                ),
                "t": torch.rand(
                    10,
                ),
            }
            for i in range(10)
        ]
    )
    extensions = _make_extensions()

    evaluator = engine.create_evaluator(
        model_with_loss, device=device, progress_bar=progress_bar
    )

    trainer = engine.create_trainer(
        model_with_loss,
        optimizer,
        20,
        device=device,
        evaluator=evaluator,
        extensions=extensions,
        out_dir=path,
    )
    trainer.run(data, data)


class Input(typing.NamedTuple):
    x: torch.Tensor
    t: torch.Tensor
    v: str


class Output(typing.NamedTuple):
    y: torch.Tensor
    loss: torch.Tensor
    v: str


class ModelNamedTupleIO(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        y = self.model(input.x)
        prefix = "train" if self.training else "val"
        loss = F.l1_loss(y, input.t)
        ppe.reporting.report({prefix + "/loss": loss})
        return Output(y, loss, input.v)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("progress_bar", [True, False])
def test_trainer_namedtuple_input(device, progress_bar, path):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = MyModel()
    ppe.to(model, device)
    model_with_loss = ModelNamedTupleIO(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [
            Input(
                torch.rand(
                    20,
                ),
                torch.rand(
                    10,
                ),
                str(i),
            )
            for i in range(10)
        ]
    )
    extensions = _make_extensions()

    evaluator = engine.create_evaluator(
        model_with_loss, device=device, progress_bar=progress_bar
    )

    trainer = engine.create_trainer(
        model_with_loss,
        optimizer,
        20,
        device=device,
        evaluator=evaluator,
        extensions=extensions,
        out_dir=path,
    )
    trainer.run(data, data)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("progress_bar", [True, False])
def test_trainer_with_code_block(device, progress_bar, path):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = MyModel()
    model_with_loss = MyModelWithLossDictOutput(model)
    ppe.to(model_with_loss, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [
            {
                "x": torch.rand(
                    20,
                ),
                "t": torch.rand(
                    10,
                ),
            }
            for i in range(10)
        ]
    )
    extensions = _make_extensions()

    evaluator = engine.create_evaluator(
        model_with_loss,
        device=device,
        progress_bar=progress_bar,
        logic=ppe.handler.CodeBlockLogic(),
    )

    trainer = engine.create_trainer(
        model_with_loss,
        optimizer,
        20,
        device=device,
        evaluator=evaluator,
        extensions=extensions,
        out_dir=path,
        logic=ppe.handler.CodeBlockLogic(),
    )
    trainer.run(data, data)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("progress_bar", [True, False])
def test_trainer_with_code_block_with_multiple_optimizers(
    device, progress_bar, path
):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = MyModel()
    model_with_loss = MyModelWithLossDictOutput(model)
    ppe.to(model_with_loss, device)
    optimizer0 = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [
            {
                "x": torch.rand(
                    20,
                ),
                "t": torch.rand(
                    10,
                ),
            }
            for i in range(10)
        ]
    )
    extensions = _make_extensions()

    evaluator = engine.create_evaluator(
        model_with_loss,
        device=device,
        progress_bar=progress_bar,
        logic=ppe.handler.CodeBlockLogic(),
    )

    trainer = engine.create_trainer(
        model_with_loss,
        {"0": optimizer0, "1": optimizer1},
        20,
        device=device,
        evaluator=evaluator,
        extensions=extensions,
        out_dir=path,
        logic=ppe.handler.CodeBlockLogic(),
    )
    trainer.run(data, data)


@pytest.mark.skipif(
    os.name == "nt" and not ppe.requires("1.9"),
    reason="torch.profiler.profile is not supported.",
)
def test_trainer_profile():
    device = "cpu"
    model = MyModel()
    model_with_loss = MyModelWithLossDictOutput(model)
    ppe.to(model_with_loss, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [
            {
                "x": torch.rand(
                    20,
                ),
                "t": torch.rand(
                    10,
                ),
            }
            for i in range(10)
        ]
    )
    extensions = _make_extensions()

    evaluator = engine.create_evaluator(model_with_loss, device=device)

    trace_handler = mock.Mock()
    warmup = 1
    active = len(data) - warmup
    profile = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        on_trace_ready=trace_handler,
        schedule=torch.profiler.schedule(wait=0, warmup=warmup, active=active),
    )
    trainer = engine.create_trainer(
        model_with_loss,
        optimizer,
        20,
        device=device,
        evaluator=evaluator,
        extensions=extensions,
        profile=profile,
    )
    trainer.run(data, data)
    assert trace_handler.call_count == 20  # n_epochs


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("progress_bar", [True, False])
def test_trainer_with_clousure_logic(device, progress_bar, path):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = MyModel()
    model_with_loss = MyModelWithLossFn(model)
    ppe.to(model_with_loss, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [
            {
                "x": torch.rand(
                    20,
                ),
                "t": torch.rand(
                    10,
                ),
            }
            for i in range(10)
        ]
    )
    extensions = _make_extensions()

    evaluator = engine.create_evaluator(
        model_with_loss,
        device=device,
        progress_bar=progress_bar,
        logic=ppe.handler.ClousureLogic(),
    )

    trainer = engine.create_trainer(
        model_with_loss,
        optimizer,
        20,
        device=device,
        evaluator=evaluator,
        extensions=extensions,
        out_dir=path,
        logic=ppe.handler.ClousureLogic(options={"backward_outputs": ["loss"]}),
    )
    trainer.run(data, data)


@pytest.mark.gpu
@pytest.mark.parametrize("autocast_train", [True, False])
@pytest.mark.parametrize("autocast_eval", [True, False])
def test_trainer_with_autocast(path, autocast_train, autocast_eval):
    if not torch.cuda.is_available():
        pytest.skip()

    class AutocastCheckModel(MyModel):
        def __init__(self, autocast_train, autocast_eval):
            super().__init__()
            self.autocast_train = autocast_train
            self.autocast_eval = autocast_eval

        def forward(self, x):
            if self.training:
                assert torch.is_autocast_enabled() == self.autocast_train
            if not self.training:
                assert torch.is_autocast_enabled() == self.autocast_eval

            return super().forward(x)

    model = AutocastCheckModel(
        autocast_train=autocast_train, autocast_eval=autocast_eval
    )
    model_with_loss = MyModelWithLossFn(model)
    ppe.to(model_with_loss, "cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [
            {
                "x": torch.rand(
                    20,
                ),
                "t": torch.rand(
                    10,
                ),
            }
            for i in range(10)
        ]
    )
    extensions = []

    evaluator = engine.create_evaluator(
        model_with_loss, device="cuda", options={"autocast": autocast_eval}
    )

    trainer = engine.create_trainer(
        model_with_loss,
        optimizer,
        20,
        device="cuda",
        evaluator=evaluator,
        extensions=extensions,
        out_dir=path,
        options={"autocast": autocast_train},
    )

    trainer.run(data, data)
