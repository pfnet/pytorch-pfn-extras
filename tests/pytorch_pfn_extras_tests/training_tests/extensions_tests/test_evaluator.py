import numpy
import pytest
import torch
import torch.distributed as dist

import pytorch_pfn_extras as ppe


class DummyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.args = []

    def forward(self, x):
        self.args.append(x)
        ppe.reporting.report({'loss': x.sum()}, self)
        return x


def custom_metric(batch, out, last_iter):
    ppe.reporting.report({'custom-metric': out.sum()})


class DummyModelTwoArgs(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.args = []

    def forward(self, x, y):
        self.args.append((x, y))
        ppe.reporting.report({'loss': x.sum() + y.sum()}, self)


def _torch_batch_to_numpy(batch):
    # In Pytorch, a batch has the batch dimension. Squeeze it for comparison.
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] == 1
    return batch.squeeze(0).numpy()


@pytest.fixture(scope='function')
def evaluator_dummies():
    data = [
        numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')
        for _ in range(2)]

    data_loader = torch.utils.data.DataLoader(data)
    target = DummyModel()
    evaluator = ppe.training.extensions.Evaluator(data_loader, target)
    expect_mean = numpy.mean([numpy.sum(x) for x in data])
    return data, data_loader, target, evaluator, expect_mean


def test_evaluate(evaluator_dummies):
    data, data_loader, target, evaluator, expect_mean = evaluator_dummies

    reporter = ppe.reporting.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        mean = evaluator.evaluate()

    # No observation is reported to the current reporter. Instead the
    # evaluator collect results in order to calculate their mean.
    assert len(reporter.observation) == 0

    assert len(target.args) == len(data)
    for i in range(len(data)):
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i]), data[i])

    numpy.testing.assert_almost_equal(
        mean['target/loss'], expect_mean, decimal=4)

    evaluator.finalize()


def test_metric(evaluator_dummies):
    data, data_loader, target, evaluator, expect_mean = evaluator_dummies
    evaluator.add_metric(custom_metric)
    mean = evaluator()
    # 'main' is used by default
    assert 'custom-metric' in mean
    numpy.testing.assert_almost_equal(
        mean['main/loss'], expect_mean, decimal=4)


def test_call(evaluator_dummies):
    data, data_loader, target, evaluator, expect_mean = evaluator_dummies

    mean = evaluator()
    # 'main' is used by default
    numpy.testing.assert_almost_equal(
        mean['main/loss'], expect_mean, decimal=4)


def test_evaluator_name(evaluator_dummies):
    data, data_loader, target, evaluator, expect_mean = evaluator_dummies

    evaluator.name = 'eval'
    mean = evaluator()
    # name is used as a prefix
    numpy.testing.assert_almost_equal(
        mean['eval/main/loss'], expect_mean, decimal=4)


def test_current_report(evaluator_dummies):
    data, data_loader, target, evaluator, expect_mean = evaluator_dummies

    reporter = ppe.reporting.Reporter()
    with reporter:
        mean = evaluator()
    # The result is reported to the current reporter.
    assert reporter.observation == mean


def test_evaluator_tuple_data():
    data = [
        (numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f'),
         numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f'))
        for _ in range(2)]

    data_loader = torch.utils.data.DataLoader(data)
    target = DummyModelTwoArgs()
    evaluator = ppe.training.extensions.Evaluator(data_loader, target)

    reporter = ppe.reporting.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        mean = evaluator.evaluate()

    assert len(target.args) == len(data)
    for i in range(len(data)):
        assert len(target.args[i]) == len(data[i])
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i][0]), data[i][0])
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i][1]), data[i][1])

    expect_mean = numpy.mean([numpy.sum(x) for x in data])
    numpy.testing.assert_almost_equal(
        mean['target/loss'], expect_mean, decimal=4)


def test_evaluator_dict_data():
    data = [
        {'x': numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f'),
         'y': numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')}
        for _ in range(2)]

    data_loader = torch.utils.data.DataLoader(data)
    target = DummyModelTwoArgs()
    evaluator = ppe.training.extensions.Evaluator(data_loader, target)

    reporter = ppe.reporting.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        mean = evaluator.evaluate()

    assert len(target.args) == len(data)
    for i in range(len(data)):
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i][0]), data[i]['x'])
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i][1]), data[i]['y'])

    expect_mean = numpy.mean(
        [numpy.sum(x['x']) + numpy.sum(x['y']) for x in data])
    numpy.testing.assert_almost_equal(
        mean['target/loss'], expect_mean, decimal=4)


def test_evaluator_with_eval_func():
    data = [
        numpy.random.uniform(-1, 1, (3, 4)).astype('f') for _ in range(2)]

    data_loader = torch.utils.data.DataLoader(data)
    target = DummyModel()
    evaluator = ppe.training.extensions.Evaluator(
        data_loader, {}, eval_func=target)

    reporter = ppe.reporting.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        evaluator.evaluate()

    assert len(target.args) == len(data)
    for i in range(len(data)):
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i]), data[i])


@pytest.fixture(scope='function')
def evaluator_with_progress():
    data = [
        numpy.random.uniform(-1, 1, (3, 4)).astype('f') for _ in range(2)]

    data_loader = torch.utils.data.DataLoader(data, batch_size=1)
    target = DummyModel()
    evaluator = ppe.training.extensions.Evaluator(
        data_loader, {}, eval_func=target, progress_bar=True)
    return evaluator, target


def test_evaluator_progress_bar(capsys, evaluator_with_progress):
    evaluator, target = evaluator_with_progress

    reporter = ppe.reporting.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        evaluator.evaluate()
    stdout = capsys.readouterr().out
    assert 'validation [.........' in stdout
    assert '         0 iterations' in stdout
    assert '       inf iters/sec.' in stdout
    assert 'validation [#########' in stdout
    assert '         1 iterations' in stdout


def test_evaluator_progress_bar_custom_label(capsys, evaluator_with_progress):
    evaluator, target = evaluator_with_progress
    evaluator.name = 'my_own_evaluator'     # Set a (long) custom name

    reporter = ppe.reporting.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        evaluator.evaluate()
    stdout = capsys.readouterr().out
    assert 'my_own_evaluator [.........' in stdout
    assert '               0 iterations' in stdout
    assert '             inf iters/sec.' in stdout
    assert 'my_own_evaluator [#########' in stdout
    assert '               1 iterations' in stdout


# Code excerpts to test IgniteEvaluator
class IgniteDummyModel(torch.nn.Module):
    def __init__(self):
        super(IgniteDummyModel, self).__init__()
        self.count = 0.

    def forward(self, *args):
        ppe.reporting.report({'x': self.count}, self)
        self.count += 1.
        return 0.


def create_dummy_evaluator(model):
    from ignite.engine import Engine

    def update_fn(engine, batch):
        y_pred = batch[1].clone().detach()
        model()
        # We return fake results for the reporters to
        # and metrics to work
        return (y_pred, y_pred)

    evaluator = Engine(update_fn)
    return evaluator


def test_ignite_evaluator_reporting_metrics():
    try:
        from ignite.metrics import MeanSquaredError
    except ImportError:
        pytest.skip('pytorch-ignite is not installed')

    # This tests verifies that either, usuer manually reported metrics
    # and ignite calculated ones are correctly reflected in the reporter
    # observation
    model = IgniteDummyModel()
    n_data = 10
    x = torch.randn((n_data, 2), requires_grad=True)
    y = torch.randn((n_data, 2))
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=3)
    evaluator = create_dummy_evaluator(model)
    # Attach metrics to the evaluator
    metric = MeanSquaredError()
    metric.attach(evaluator, 'mse')
    evaluator_ignite_ext = ppe.training.extensions.IgniteEvaluator(
        evaluator, loader, model, progress_bar=False
    )
    reporter = ppe.reporting.Reporter()
    with reporter:
        result = evaluator_ignite_ext()
    # Internally reported metrics
    assert result['main/x'] == 1.5
    # Ignite calculated metric
    assert result['val/mse'] == 0.0


def test_distributed_evaluation(mocker):
    mocker.patch.object(dist, 'is_initialized', return_value=True)

    dummy_data = []   # Note: has no effect to the evaluation
    data_loader = torch.utils.data.DataLoader(dummy_data)
    target = DummyModel()
    evaluator = ppe.training.extensions.DistributedEvaluator(data_loader, target)

    # Make a (simulated) summary for each rank
    worker_evaluations = [
        [1., 2., 3.],   # assuming rank=0
        [4., 5., 6.],   # rank=1
        [7., 8., 9.],   # ...
        [10., 11., 12.],
    ]
    worker_summaries = []
    for accs in worker_evaluations:
        s = ppe.reporting.DictSummary()
        for acc in accs:
            s.add({'target/score': acc})
        worker_summaries.append(s)

    mocker.patch.object(ppe.training.extensions.evaluator, '_dist_gather',
                        return_value=worker_summaries)

    reporter = ppe.reporting.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        mean = evaluator.evaluate()
        assert mean['target/score'] == 6.5


def test_distributed_evaluator_progress_bar(mocker):
    mocker.patch.object(dist, 'is_initialized', return_value=True)

    data_loader = torch.utils.data.DataLoader([])
    target = DummyModel()

    mocker.patch.object(dist, 'get_rank', return_value=0)
    evaluator = ppe.training.extensions.DistributedEvaluator(data_loader, target, progress_bar=True)
    assert evaluator._progress_bar

    # rank != 0 will forcibly set progress_bar=False
    mocker.patch.object(dist, 'get_rank', return_value=1)
    evaluator = ppe.training.extensions.DistributedEvaluator(data_loader, target, progress_bar=True)
    assert not evaluator._progress_bar
