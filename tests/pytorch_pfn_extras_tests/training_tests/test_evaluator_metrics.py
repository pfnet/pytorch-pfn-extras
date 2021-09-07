import pytest

import torch

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras import engine


class MyModel(torch.nn.Module):
    def __init__(self, correct_ratio):
        super().__init__()
        self.correct_ratio = correct_ratio

    def forward(self, x, t):
        g = t.clone()
        to_alter = int(10 * (1 - self.correct_ratio))
        g[0:to_alter][:] -= 1
        return {'y': g}


@pytest.mark.parametrize('device', ['cpu'])
@pytest.mark.parametrize('accuracy', [0, 0.5, 1.0])
def test_evaluator_with_metric(device, accuracy):
    model = MyModel(accuracy)
    data = torch.utils.data.DataLoader(
        [{'x': torch.rand(20), 't': torch.rand(1)} for i in range(10)],
        batch_size=10)

    evaluator = engine.create_evaluator(
        model, device=device,
        metrics=[ppe.training.metrics.AccuracyMetric('t', 'y')],
        handler_options={'eval_report_keys': ['accuracy']})
    evaluator.handler.eval_setup(evaluator, data)
    reporter = ppe.reporting.Reporter()
    observation = {}
    with reporter.scope(observation):
        evaluator.run(data)
    assert pytest.approx(observation['val/accuracy'], accuracy)


class AsyncModel(torch.nn.Module):
    def __init__(self, correct_ratio):
        super().__init__()
        self.correct_ratio = correct_ratio
        self._current_it = 0
        self._outs = []
        self._pending_called = False

    def forward(self, x, t):
        g = t.clone()
        to_alter = int(10 * (1 - self.correct_ratio))
        g[0:to_alter][:] -= 1
        self._outs.append({'y': g})
        return {}

    def get_pending_out(self, block):
        # Retrieve the out once every 4 times if block == False
        self._pending_called = True
        self._current_it += 1
        out = None
        if block or self._current_it % 4 == 0:
            out, self._outs = self._outs[0], self._outs[1:]
        return out


class DeferRuntime(ppe.runtime.PyTorchRuntime):
    def move_module(self, module):
        return module.to('cpu')

    def move_tensor(self, tensor):
        return tensor.to('cpu')

    def get_pending_result(self, module, block):
        return module.get_pending_out(block)


@pytest.mark.parametrize('accuracy', [0, 0.5, 1.0])
def test_evaluator_async(accuracy):
    device = 'async-cpu'
    model = AsyncModel(accuracy)
    data = torch.utils.data.DataLoader(
        [{'x': torch.rand(20), 't': torch.rand(1)} for i in range(1000)],
        batch_size=10)

    options = {'eval_report_keys': ['accuracy'],
               'async': True}
    # Register the handler
    ppe.runtime.runtime_registry.register(device, DeferRuntime)

    ppe.to(model, device)
    evaluator = engine.create_evaluator(
        model, device=device, handler_options=options,
        metrics=[ppe.training.metrics.AccuracyMetric('t', 'y')])

    reporter = ppe.reporting.Reporter()
    observation = {}
    with reporter.scope(observation):
        evaluator.run(data)
    assert pytest.approx(observation['val/accuracy'], accuracy)
    assert model._pending_called
