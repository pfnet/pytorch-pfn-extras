import tempfile
import typing

import pytest

import torch
from torch import nn
from torch.nn import functional as F
from unittest import mock

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


def _make_extensions():
    return [
        training.extensions.LogReport(trigger=(10, 'iteration')),
        training.extensions.ProgressBar(update_interval=2),
        training.extensions.PrintReport(
            [
                'epoch',
                'iteration',
                'train/loss',
                'val/loss',
                'val/accuracy',
                'elapsed_time',
                'time',
            ]
        ),
    ]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_trainer(device, path):
    model = MyModel()
    ppe.to(model, device)
    model_with_loss = MyModelWithLossFn(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [(torch.rand(20,), torch.rand(10,)) for i in range(10)])
    extensions = _make_extensions()

    trainer = engine.create_trainer(
        model_with_loss, optimizer, 20,
        device=device, extensions=extensions,
        out_dir=path,
    )
    trainer.run(data, data)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('progress_bar', [True, False])
def test_train_with_evaluator(device, progress_bar, path):
    model = MyModel()
    ppe.to(model, device)
    model_with_loss = MyModelWithLossFn(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [(torch.rand(20,), torch.rand(10,)) for i in range(10)])
    extensions = _make_extensions()

    evaluator = engine.create_evaluator(
        model_with_loss, device=device, progress_bar=progress_bar)

    trainer = engine.create_trainer(
        model_with_loss, optimizer, 20,
        device=device, evaluator=evaluator, extensions=extensions,
        out_dir=path
    )
    mpath = 'pytorch_pfn_extras.training._evaluator._Evaluator.run'
    with mock.patch(mpath) as patched:
        trainer.run(data, data)
        assert patched.call_count == 20


@pytest.mark.parametrize(
    "evaluator_trigger",
    [(20, (1, "epoch")), (40, (5, "iteration"))],
)
def test_evaluator_trigger(evaluator_trigger, path):
    device = 'cpu'
    progress_bar = False
    model = MyModel()
    ppe.to(model, device)
    model_with_loss = MyModelWithLossFn(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [(torch.rand(20,), torch.rand(10,)) for i in range(10)])
    extensions = _make_extensions()

    evaluator = engine.create_evaluator(
        model_with_loss, device=device, progress_bar=progress_bar)

    trainer = engine.create_trainer(
        model_with_loss, optimizer, 20,
        device=device, evaluator=(evaluator, evaluator_trigger[1]),
        extensions=extensions, out_dir=path
    )
    path = 'pytorch_pfn_extras.training._evaluator._Evaluator.run'
    with mock.patch(path) as patched:
        trainer.run(data, data)
        assert patched.call_count == evaluator_trigger[0]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_train_result_equal(device, path):
    train_data = torch.utils.data.DataLoader(
        [(torch.rand(20,), torch.rand(10,)) for i in range(10)])
    eval_data = torch.utils.data.DataLoader(
        [(torch.rand(20,), torch.rand(10,)) for i in range(5)])
    data = torch.utils.data.DataLoader(
        [(torch.rand(20,),) for i in range(30)])

    def get_result_from_trainer():
        model = MyModel()
        ppe.to(model, device)
        model_with_loss = MyModelWithLossFn(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        extensions = _make_extensions()

        trainer = engine.create_trainer(
            model_with_loss, optimizer, 20,
            device=device, extensions=extensions,
            out_dir=path
        )
        trainer.run(train_data, eval_data)

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


class DeferRuntime(ppe.runtime.PyTorchRuntime):
    def move_module(self, module):
        return module.to('cpu')

    def move_tensor(self, tensor):
        return tensor.to('cpu')

    def get_pending_result(self, module, block):
        return module.get_pending_out(block)


class MyModelWithLossAsync(MyModelWithLossFn):
    def __init__(self, model):
        super().__init__(model)
        self._current_it = 0
        self._outs = []
        self._pending_called = False

    def forward(self, x, t):
        self._outs.append(super().forward(x, t))

    def get_pending_out(self, block):
        # Retrieve the out once every 4 times if block == False
        self._pending_called = True
        self._current_it += 1
        out = None
        if block or self._current_it % 4 == 0:
            out, self._outs = self._outs[0], self._outs[1:]
        return out


def test_trainer_defer(path):
    class Extension:
        def __init__(self, is_async):
            self.name = 'Dummy'
            self.trigger = (1, 'iteration')
            self.called = 0
            self.is_async = is_async

        def __call__(self, manager):
            self.called += 1

    device = 'async-cpu'
    model = MyModel()
    model_with_loss = MyModelWithLossAsync(model)
    # Register the handler
    ppe.runtime.runtime_registry.register(device, DeferRuntime)
    ppe.to(model_with_loss, device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [(torch.rand(20,), torch.rand(10,)) for i in range(100)])

    extensions = [Extension(True), Extension(False)]
    options = {'async': True}

    trainer = engine.create_trainer(
        model_with_loss, optimizer, 2, options=options,
        device=device, extensions=extensions,
        out_dir=path
    )
    trainer.run(data, data)
    assert trainer.manager.iteration == 200
    assert trainer.manager.execution == 200
    assert extensions[0].called == 200
    assert extensions[1].called == 200
    assert model_with_loss._pending_called


def test_trainer_defer_wrong_order(path):
    class WrongOrderHandler(ppe.handler.Handler):
        def _complete_train_step(self, trainer, outs, block, sn, sm, rt):
            idx, batch, cback = self.pending_iters[sn][0]
            if idx < 10:
                super()._complete_train_step(trainer, outs, block, sn, sm, rt)
            else:
                cback(90, None, is_deferred=block)

    device = 'async-cpu'
    model = MyModel()
    model_with_loss = MyModelWithLossAsync(model)
    # Register the handler
    ppe.runtime.runtime_registry.register(device, DeferRuntime)
    ppe.to(model_with_loss, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [(torch.rand(20,), torch.rand(10,)) for i in range(100)])

    options = {'async': True}

    trainer = engine.create_trainer(
        model_with_loss, optimizer, 2, options=options,
        device=device, handler_class=WrongOrderHandler,
        out_dir=path
    )
    with pytest.raises(RuntimeError, match="Completed a not expected"):
        trainer.run(data, data)


def _compare_states(s1, s2):
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
            all_equal = all_equal and torch.allclose(s1[k], s2[k])
        else:
            all_equal = all_equal and s1[k] == s2[k]
        if not all_equal:
            return all_equal
    return all_equal


class TestTrainerState:
    def _get_trainer(self, epochs, out_dir):
        model = MyModel()
        ppe.to(model, 'cpu')
        model_with_loss = MyModelWithLossFn(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        extensions = _make_extensions()
        trainer = engine.create_trainer(
            model_with_loss, optimizer, 20,
            device='cpu', extensions=extensions,
            out_dir=out_dir
        )
        return trainer

    def test_trainer_state(self, path):
        torch.manual_seed(0)
        trainer = self._get_trainer(20, path)
        data = torch.utils.data.DataLoader(
            [(torch.ones(20,), torch.ones(10,)) for i in range(10)])
        trainer.run(data, data)
        # State to be compared to
        state = trainer.state_dict()
        torch.manual_seed(0)
        new_trainer = self._get_trainer(10, path)
        new_trainer.run(data, data)
        assert not _compare_states(state, new_trainer.state_dict())
        new_trainer = self._get_trainer(20, path)
        new_trainer.load_state_dict(trainer.state_dict())
        new_trainer.run(data, data)
        assert _compare_states(state, new_trainer.state_dict())

    def test_trainer_autoload(self, path):
        trainer = self._get_trainer(20, path)
        data = torch.utils.data.DataLoader(
            [(torch.rand(20,), torch.rand(10,)) for i in range(10)])
        trainer.extend(ppe.training.extensions.snapshot())
        trainer.run(data, data)

        new_trainer = self._get_trainer(20, path)
        new_trainer.extend(ppe.training.extensions.snapshot(autoload=True))
        # This forces engine initialization
        new_trainer._setup_manager(len(data))
        assert new_trainer.epoch == 20
        assert _compare_states(
            trainer.state_dict(), new_trainer.state_dict())


class MyModelWithLossDictOutput(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t):
        y = self.model(x)
        prefix = 'train' if self.training else 'val'
        loss = F.l1_loss(y, t)
        ppe.reporting.report({prefix + '/loss': loss})
        return {'y': y, 'loss': loss}


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('progress_bar', [True, False])
def test_trainer_dict_input(device, progress_bar, path):
    model = MyModel()
    ppe.to(model, device)
    model_with_loss = MyModelWithLossDictOutput(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [{'x': torch.rand(20,), 't': torch.rand(10,)} for i in range(10)])
    extensions = _make_extensions()

    evaluator = engine.create_evaluator(
        model_with_loss, device=device, progress_bar=progress_bar)

    trainer = engine.create_trainer(
        model_with_loss, optimizer, 20,
        device=device, evaluator=evaluator, extensions=extensions,
        out_dir=path
    )
    trainer.run(data, data)


class Input(typing.NamedTuple):
    x: torch.Tensor
    t: torch.Tensor


class Output(typing.NamedTuple):
    y: torch.Tensor
    loss: torch.Tensor


class MyModelWithLossNamedTupleOutput(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        y = self.model(input.x)
        prefix = 'train' if self.training else 'val'
        loss = F.l1_loss(y, input.t)
        ppe.reporting.report({prefix + '/loss': loss})
        return Output(y, loss)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('progress_bar', [True, False])
def test_trainer_namedtuple_input(device, progress_bar, path):
    model = MyModel()
    ppe.to(model, device)
    model_with_loss = MyModelWithLossNamedTupleOutput(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    data = torch.utils.data.DataLoader(
        [Input(torch.rand(20,), torch.rand(10,)) for i in range(10)])
    extensions = _make_extensions()

    evaluator = engine.create_evaluator(
        model_with_loss, device=device, progress_bar=progress_bar)

    trainer = engine.create_trainer(
        model_with_loss, optimizer, 20,
        device=device, evaluator=evaluator, extensions=extensions,
        out_dir=path
    )
    trainer.run(data, data)
