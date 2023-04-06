import unittest.mock
import contextlib

import torch
import pytest

import pytorch_pfn_extras as ppe


def torch_testing_assert_close(*args, **kwargs):
    if ppe.requires("1.10.0"):
        torch.testing.assert_close(*args, **kwargs)
    else:
        torch.testing.assert_allclose(*args, **kwargs)


class MockRuntime(ppe.runtime.BaseRuntime):
    def __init__(self, device, options):
        super().__init__(device, options)
        self._initialize_called = False
        self._train_epoch_begin_called = False
        self._called_module = None

    def move_module(self, module):
        return module

    def move_tensor(self, tensor):
        return tensor

    def convert_batch(self, batch):
        class BatchWrapper:
            def __init__(self, batch):
                self.batch = batch
                self.converted = True

        return BatchWrapper(batch)

    def initialize_module(self, module, loader_or_sample, optimizer):
        assert loader_or_sample is not None
        self._initialize_called = True
        self._called_module = module

    def train_epoch_begin(self, module):
        self._train_epoch_begin_called = True
        self._called_module = module

    def train_epoch_end(self, module):
        self._train_epoch_end_called = True
        self._called_module = module

    def train_cleanup(self, module):
        self._train_cleanup_called = True
        self._called_module = module

    def train_pre_step(self, trainer, module, batch_idx, batch):
        self._train_pre_step_called = True
        self._called_module = module

    def train_post_step(self, trainer, module, batch_idx, batch, outputs):
        self._train_post_step_called = True
        self._called_module = module

    def train_validation_begin(self, module):
        self._train_validation_begin_called = True
        self._called_module = module

    def eval_pre_step(self, evaluator, module, batch_idx, batch):
        self._eval_pre_step_called = True
        self._called_module = module

    def eval_post_step(self, evaluator, module, batch_idx, batch, outputs):
        self._eval_post_step_called = True
        self._called_module = module

    @classmethod
    @contextlib.contextmanager
    def trace(cls, event_name, arg):
        arg["called"] = True
        yield


class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sm1 = torch.nn.Linear(1, 1)
        self.sm2 = torch.nn.Linear(1, 1)

    def forward(self, x):
        return 1


class MockTrainer:
    def __init__(self):
        self.models = {'main': MockModule()}
        self.optimizers = {}
        self.epoch = 0


class MockEvaluator:
    def __init__(self):
        self.models = {'main': MockModule()}
        self.optimizers = {}
        self.epoch = 0


class MockLogic(ppe.handler.BaseLogic):
    def train_epoch_begin(self, epoch, models, loader):
        self._train_epoch_begin_called = True

    def train_epoch_end(self, epoch, models):
        self._train_epoch_end_called = True

    def train_step(self, models, optimizers, batch_idx, batch):
        assert batch.converted
        return models['main'](batch)

    def train_step_optimizers(self, models, optimizers, batch_idx):
        self._train_step_optimizers_called = True

    def train_validation_begin(self, models):
        self._train_validation_begin_called = True

    def eval_step(self, models, batch_idx, batch):
        assert batch.converted
        return models['main'](batch)


class HandlerTester:
    def _get_handler(self, options=None):
        if options is None:
            options = {}
        ppe.runtime.runtime_registry.register('test_rt', MockRuntime)
        trainer = MockTrainer()
        logic = MockLogic()
        handler = ppe.handler.Handler(
            logic, MockRuntime('test_rt', {}), options
        )
        return handler, trainer, logic

    def _move_modules(self, module, to_move):
        for name in to_move:
            if name == 'self':
                ppe.to(module, 'test_rt')
            else:
                ppe.to(getattr(module, name), 'test_rt')

    def _assert_called(self, module, to_move, function):
        for name, mod in module.named_modules():
            if (mod is module and 'self' in to_move) or (name in to_move):
                assert getattr(mod._ppe_runtime, f'_{function}_called')
                assert mod._ppe_runtime._called_module == mod
            else:
                if hasattr(mod, '_ppe_runtime'):
                    assert mod._ppe_runtime._called_module != mod


class TestHandlerTrainSync(HandlerTester):

    @pytest.mark.parametrize(
        'to_move', [('self',), ('sm1',), ('sm2',), ('sm1', 'sm2')]
    )
    def test_train_setup(self, to_move):
        handler, trainer, _ = self._get_handler()
        module = trainer.models['main']
        self._move_modules(module, to_move)
        handler.train_setup(trainer, [])
        self._assert_called(module, to_move, 'initialize')

    @pytest.mark.parametrize(
        'to_move', [('self',), ('sm1',), ('sm2',), ('sm1', 'sm2')]
    )
    def test_train_cleanup(self, to_move):
        handler, trainer, _ = self._get_handler()
        module = trainer.models['main']
        self._move_modules(module, to_move)
        handler.train_cleanup(trainer)
        self._assert_called(module, to_move, 'train_cleanup')

    @pytest.mark.parametrize(
        'to_move', [('self',), ('sm1',), ('sm2',), ('sm1', 'sm2')]
    )
    def test_train_epoch_begin(self, to_move):
        handler, trainer, logic = self._get_handler()
        module = trainer.models['main']
        self._move_modules(module, to_move)
        handler.train_epoch_begin(trainer, [])
        self._assert_called(module, to_move, 'train_epoch_begin')
        assert logic._train_epoch_begin_called

    @pytest.mark.parametrize(
        'to_move', [('self',), ('sm1',), ('sm2',), ('sm1', 'sm2')]
    )
    def test_train_epoch_end(self, to_move):
        handler, trainer, logic = self._get_handler()
        module = trainer.models['main']
        self._move_modules(module, to_move)
        # Should check that the handler completes
        handler.train_epoch_end(trainer)
        self._assert_called(module, to_move, 'train_epoch_end')
        assert logic._train_epoch_end_called

    @pytest.mark.parametrize(
        'to_move', [('self',), ('sm1',), ('sm2',), ('sm1', 'sm2')]
    )
    def test_train_step(self, to_move):
        handler, trainer, logic = self._get_handler()
        module = trainer.models['main']
        self._move_modules(module, to_move)

        callback = unittest.mock.Mock(return_value=None)
        handler.train_step(trainer, 0, None, callback)
        callback.assert_called_once_with(0, 1)

        self._assert_called(module, to_move, 'train_pre_step')
        assert logic._train_step_optimizers_called

    @pytest.mark.parametrize(
        'to_move', [('self',), ('sm1',), ('sm2',), ('sm1', 'sm2')]
    )
    def test_train_post_step(self, to_move):
        options = {'train_report_keys': ['output']}
        handler, trainer, _ = self._get_handler(options)
        module = trainer.models['main']
        self._move_modules(module, to_move)
        reporter = ppe.reporting.Reporter()
        with reporter:
            handler.train_post_step(trainer, 0, None, {'output': 1})
        assert reporter.observation['train/output'] == 1
        self._assert_called(module, to_move, 'train_post_step')


class TestHandlerValidationSync(HandlerTester):
    def _get_handler(self, options=None):
        handler, _, logic = super()._get_handler(options)
        evaluator = MockEvaluator()
        return handler, evaluator, logic

    @pytest.mark.parametrize(
        'to_move', [('self',), ('sm1',), ('sm2',), ('sm1', 'sm2')]
    )
    def test_eval_setup(self, to_move):
        handler, evaluator, _ = self._get_handler()
        module = evaluator.models['main']
        self._move_modules(module, to_move)
        handler.eval_setup(evaluator, [])
        self._assert_called(module, to_move, 'initialize')

    @pytest.mark.parametrize(
        'to_move', [('self',), ('sm1',), ('sm2',), ('sm1', 'sm2')]
    )
    def test_train_validation_begin(self, to_move):
        handler, evaluator, logic = self._get_handler()
        module = evaluator.models['main']
        self._move_modules(module, to_move)
        handler.train_validation_begin(None, evaluator)
        self._assert_called(module, to_move, 'train_validation_begin')
        assert logic._train_validation_begin_called

    @pytest.mark.parametrize(
        'to_move', [('self',), ('sm1',), ('sm2',), ('sm1', 'sm2')]
    )
    def test_eval_step(self, to_move):
        handler, evaluator, logic = self._get_handler()
        module = evaluator.models['main']
        self._move_modules(module, to_move)

        callback = unittest.mock.Mock(return_value=None)
        handler.eval_step(evaluator, 0, None, callback)
        callback.assert_called_once_with(0, 1)

        self._assert_called(module, to_move, 'eval_pre_step')

    @pytest.mark.parametrize(
        'to_move', [('self',), ('sm1',), ('sm2',), ('sm1', 'sm2')]
    )
    def test_train_post_step(self, to_move):
        options = {'eval_report_keys': ['output']}
        handler, evaluator, _ = self._get_handler(options)
        module = evaluator.models['main']
        self._move_modules(module, to_move)
        reporter = ppe.reporting.Reporter()
        with reporter:
            handler.eval_post_step(evaluator, 0, None, {'output': 1})
        assert reporter.observation['val/output'] == 1
        self._assert_called(module, to_move, 'eval_post_step')


@pytest.mark.gpu
class TestHandlerAutocast:
    def run_autocast(self, options):
        trainer = MockTrainer()
        logic = ppe.handler.Logic(options=options)
        handler = ppe.handler.Handler(
            logic, ppe.runtime.PyTorchRuntime('cuda', {}), {}
        )

        completed = False

        class _MModule(torch.nn.Module):
            def forward(self, x, y):
                return torch.mm(x, y)

        trainer.models['main'] = _MModule()
        trainer.optimizers['main'] = torch.optim.SGD(
            [torch.nn.Parameter(torch.zeros(10))], 0.01
        )
        ppe.to(trainer.models['main'], 'cuda')
        completed = False

        def callback(batch_idx, outs):
            nonlocal completed
            if "autocast" in options:
                if isinstance(options["autocast"], bool):
                    if options["autocast"]:
                        assert outs.dtype == torch.float16
                    else:
                        assert outs.dtype == torch.float32
                else:
                    if options["autocast"]["device_type"] == "cuda":
                        assert outs.dtype == options["autocast"]["dtype"]
                    # For some reason result for CPU is float32 instead of bfloat16
            completed = True

        inputs = {
            'x': torch.rand((2, 2)).cuda(),
            'y': torch.rand((2, 2)).cuda(),
        }
        handler.train_step(trainer, 0, inputs, callback)
        assert completed

    @pytest.mark.parametrize('autocast', [True, False])
    def test_autocast(self, autocast):
        self.run_autocast({'autocast': autocast})

    def test_autocast_not_enabled(self):
        old_enable = ppe.runtime._autocast._cuda_amp_available
        try:
            ppe.runtime._autocast._cuda_amp_available = False
            with pytest.raises(RuntimeError):
                ppe.handler.Logic(options={'autocast': True})
        finally:
            ppe.runtime._autocast._cuda_amp_available = old_enable

    @pytest.mark.skipif(not ppe.requires("1.10.0"), reason="requires PyTorch>=1.10")
    @pytest.mark.parametrize(
        'device_type, dtype',
        [("cpu", torch.bfloat16), ("cuda", torch.float16)]
    )
    def test_autocast_options(self, device_type, dtype):
        self.run_autocast(
            {"autocast": {'device_type': device_type, "dtype": dtype}}
        )


class TestLogic:

    def test_train_epoch_begin(self):
        # Check that the DataLoader has the sampler updated
        class _MockedDL:
            def __init__(self):
                class _Sampler:
                    def set_epoch(self, epoch):
                        self.epoch = epoch
                self.sampler = _Sampler()
        logic = ppe.handler.Logic()
        loader = _MockedDL()
        models = {'main': torch.nn.Linear(1, 1)}
        # The model should be set to train mode
        models['main'].eval()
        assert not models['main'].training
        logic.train_epoch_begin(models, 10, loader)
        assert models['main'].training
        assert loader.sampler.epoch == 10

    def _run_step(self, logic, device):
        input = torch.rand(1, 1).to(device)
        input.requires_grad = True

        class _Module(torch.nn.Linear):
            def __init__(self) -> None:
                super().__init__(1, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return super().forward(x).sum()

        model = _Module().to(device)
        models = {'main': model}
        optimizers = {'main': torch.optim.SGD(model.parameters(), 1.0, 0)}
        out = logic.train_step(models, optimizers, 0, input)
        return models, optimizers, input, out

    def test_train_step(self):
        logic = ppe.handler.Logic()
        models, optimizers, input, out = self._run_step(logic, 'cpu')
        model = models['main']
        assert input.grad is not None
        # The gradient of a linear layer is its transposed weight
        torch_testing_assert_close(input.grad, model.weight.T)
        torch_testing_assert_close(out, model(input))

    @pytest.mark.parametrize(
        'to_backprop',
        [None, ('0',), ('0', '1'), ('0', '1', '2'), ('1', '2'), ('2',)]
    )
    def test_train_step_backward(self, to_backprop):
        logic = ppe.handler.Logic(options={'backward_outputs': to_backprop})
        input = torch.rand(1, 1)
        input.requires_grad = True

        class _MultiOutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l0 = torch.nn.Linear(1, 1)
                self.l1 = torch.nn.Linear(1, 1)
                self.l2 = torch.nn.Linear(1, 1)

            def forward(self, x):
                return {'0': self.l0(x), '1': self.l1(x), '2': self.l2(x)}
        model = _MultiOutModel()
        models = {'main': model}
        optimizers = {'main': torch.optim.SGD(model.parameters(), 1.0)}
        assert input.grad is None

        if to_backprop is None:
            to_backprop = ('0', '1', '2')

        # Copy the original parameters to check that they were not updated
        original_parameters = {}
        for val in to_backprop:
            original_parameters[val] = getattr(
                model, f'l{val}').weight.detach().clone()

        outs = logic.train_step(models, optimizers, 0, input)

        assert isinstance(outs, dict)
        assert len(outs.keys()) == 3
        grad = torch.zeros(1)
        for val in to_backprop:
            grad = grad + getattr(model, f'l{val}').weight.T
        torch_testing_assert_close(input.grad, grad)

        # Check that logic step does not change the value of weight
        for val in original_parameters:
            torch_testing_assert_close(
                original_parameters[val], getattr(model, f'l{val}').weight)

    def test_train_step_backward_nograd(self):
        logic = ppe.handler.Logic()
        input = torch.rand(1, 1)
        input.requires_grad = True

        class _DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l0 = torch.nn.Linear(1, 1)

            def forward(self, x):
                return {'0': x}

        model = _DummyModel()
        models = {'main': model}
        optimizers = {'main': torch.optim.SGD(model.parameters(), 1.0)}
        assert input.grad is None

        outs = logic.train_step(models, optimizers, 0, input)

        assert outs['0'].grad is None

    def test_train_step_backward_invalid(self):
        logic = ppe.handler.Logic(options={'backward_outputs': 'abcd'})
        input = torch.rand(1, 1)
        input.requires_grad = True

        class _DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l0 = torch.nn.Linear(1, 1)

            def forward(self, x):
                return {'0': x}

        model = _DummyModel()
        models = {'main': model}
        optimizers = {'main': torch.optim.SGD(model.parameters(), 1.0)}
        assert input.grad is None

        with pytest.warns(UserWarning, match='backward value: abcd'):
            logic.train_step(models, optimizers, 0, input)

    def test_train_step_optimizers(self):
        logic = ppe.handler.Logic()
        models, optimizers, input, out = self._run_step(logic, 'cpu')
        model = models['main']
        m_weight = model.weight.clone().detach()
        w_grad = model.weight.grad.clone().detach()
        logic.train_step_optimizers(model, optimizers, 0)
        # Checks that the value was correctly updated
        torch_testing_assert_close(m_weight - w_grad, model.weight.T)

    @pytest.mark.gpu
    def test_grad_scaler(self):
        scaler = torch.cuda.amp.GradScaler()
        options = {'grad_scaler': scaler}
        logic = ppe.handler.Logic(options=options)
        models, optimizers, input, out = self._run_step(logic, 'cuda')
        model = models['main']
        m_weight = model.weight.clone().detach()
        w_grad = model.weight.grad.clone().detach()
        # The gradient of a linear layer is its transposed weight
        torch_testing_assert_close(input.grad, scaler.scale(model.weight.T))
        torch_testing_assert_close(out, model(input))
        logic.train_step_optimizers(model, optimizers, 0)
        # Checks that the value was correctly updated and gradients deescaled
        # before the update
        torch_testing_assert_close(
            scaler.scale(m_weight) - w_grad, scaler.scale(model.weight.T))

    @pytest.mark.gpu
    def test_invalid_grad_scaler(self):
        options = {'grad_scaler': object()}
        with pytest.raises(RuntimeError):
            ppe.handler.Logic(options=options)

    @pytest.mark.gpu
    def test_disabled_grad_scaler(self):
        old_enable = ppe.runtime._autocast._cuda_amp_available
        try:
            ppe.runtime._autocast._cuda_amp_available = False
            options = {'grad_scaler': torch.cuda.amp.GradScaler()}
            with pytest.raises(RuntimeError):
                ppe.handler.Logic(options=options)
        finally:
            ppe.runtime._autocast._cuda_amp_available = old_enable

    def test_train_validation_begin(self):
        logic = ppe.handler.Logic()
        models = {'main': torch.nn.Linear(1, 1)}
        models['main'].train()
        assert models['main'].training
        logic.train_validation_begin(models)
        assert not models['main'].training

    def test_eval_step(self):
        logic = ppe.handler.Logic()
        input = torch.rand(1, 1)
        model = torch.nn.Linear(1, 1)
        models = {'main': model}
        models['main'].eval()
        out = logic.eval_step(models, 0, input)
        torch_testing_assert_close(out, model(input))

    @pytest.mark.gpu
    def test_use_grad_scaler_with_clousure(self):
        options = {'grad_scaler': torch.cuda.amp.GradScaler()}
        with pytest.raises(RuntimeError):
            ppe.handler.ClousureLogic(options=options)
