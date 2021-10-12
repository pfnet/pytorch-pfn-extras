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
        options={'eval_report_keys': ['accuracy']})
    evaluator.handler.eval_setup(evaluator, data)
    reporter = ppe.reporting.Reporter()
    observation = {}
    with reporter.scope(observation):
        evaluator.run(data)
    assert pytest.approx(observation['val/accuracy'], accuracy)


class AsyncResult(ppe.handler.DeferredResult):
    def __init__(self, value):
        self._period = 4
        self._count = 0
        self._done = False
        self._value = value

    def done(self):
        self._count += 1
        if self._count == self._period:
            self._done = True
        return self._done

    def wait(self):
        self._done = True

    def get(self):
        if self._done or (self._count == self._period):
            return self._value
        return None


class AsyncModel(torch.nn.Module):
    def __init__(self, correct_ratio):
        super().__init__()
        self.correct_ratio = correct_ratio
        self._current_it = 0

    def forward(self, x, t):
        g = t.clone()
        to_alter = int(10 * (1 - self.correct_ratio))
        g[0:to_alter][:] -= 1
        outs = {'y': g}
        return AsyncResult(outs)


@pytest.mark.parametrize('accuracy', [0, 0.5, 1.0])
def test_evaluator_async(accuracy):
    device = 'cpu'
    model = AsyncModel(accuracy)
    data = torch.utils.data.DataLoader(
        [{'x': torch.rand(20), 't': torch.rand(1)} for i in range(1000)],
        batch_size=10)

    options = {'eval_report_keys': ['accuracy']}

    ppe.to(model, device)
    evaluator = engine.create_evaluator(
        model, device=device, options=options,
        metrics=[ppe.training.metrics.AccuracyMetric('t', 'y')])

    reporter = ppe.reporting.Reporter()
    observation = {}
    with reporter.scope(observation):
        evaluator.run(data)
    assert pytest.approx(observation['val/accuracy'], accuracy)
