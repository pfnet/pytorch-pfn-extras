import io
import math
import tempfile
import threading
import time

import numpy
import pytest
import torch

import pytorch_pfn_extras as ppe


def test_empty_reporter():
    reporter = ppe.reporting.Reporter()
    assert reporter.observation == {}


def test_enter_exit():
    reporter1 = ppe.reporting.Reporter()
    reporter2 = ppe.reporting.Reporter()
    with reporter1:
        assert ppe.reporting.get_current_reporter() is reporter1
        with reporter2:
            assert ppe.reporting.get_current_reporter() is reporter2
        assert ppe.reporting.get_current_reporter() is reporter1


def test_enter_exit_threadsafe():
    # This test ensures reporter.__enter__ correctly stores the reporter
    # in the thread-local storage.

    def thread_func(reporter, record):
        with reporter:
            # Sleep for a tiny moment to cause an overlap of the context
            # managers.
            time.sleep(0.01)
            record.append(ppe.reporting.get_current_reporter())

    record1 = []  # The current reporter in each thread is stored here.
    record2 = []
    reporter1 = ppe.reporting.Reporter()
    reporter2 = ppe.reporting.Reporter()
    thread1 = threading.Thread(
        target=thread_func,
        args=(reporter1, record1))
    thread2 = threading.Thread(
        target=thread_func,
        args=(reporter2, record2))
    thread1.daemon = True
    thread2.daemon = True
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    assert record1[0] is reporter1
    assert record2[0] is reporter2


def test_scope():
    reporter1 = ppe.reporting.Reporter()
    reporter2 = ppe.reporting.Reporter()
    with reporter1:
        observation = {}
        with reporter2.scope(observation):
            assert ppe.reporting.get_current_reporter() is reporter2
            assert reporter2.observation is observation
        assert ppe.reporting.get_current_reporter() is reporter1
        assert reporter2.observation is not observation


def test_add_observer():
    reporter = ppe.reporting.Reporter()
    observer = object()
    reporter.add_observer('o', observer)

    reporter.report({'x': 1}, observer)

    observation = reporter.observation
    assert 'o/x' in observation
    assert observation['o/x'] == 1
    assert 'x'not in observation


def test_add_observers():
    reporter = ppe.reporting.Reporter()
    observer1 = object()
    reporter.add_observer('o1', observer1)
    observer2 = object()
    reporter.add_observer('o2', observer2)

    reporter.report({'x': 1}, observer1)
    reporter.report({'y': 2}, observer2)

    observation = reporter.observation
    assert 'o1/x' in observation
    assert observation['o1/x'] == 1
    assert 'o2/y' in observation
    assert observation['o2/y'] == 2
    assert 'x' not in observation
    assert 'y' not in observation
    assert 'o1/y' not in observation
    assert 'o2/x' not in observation


def test_report_without_observer():
    reporter = ppe.reporting.Reporter()
    reporter.report({'x': 1})

    observation = reporter.observation
    assert 'x' in observation
    assert observation['x'] == 1


# ppe.reporting.report

def test_report_without_reporter():
    observer = object()
    ppe.reporting.report({'x': 1}, observer)


def test_report():
    reporter = ppe.reporting.Reporter()
    with reporter:
        ppe.reporting.report({'x': 1})
    observation = reporter.observation
    assert 'x' in observation
    assert observation['x'] == 1


def test_report_with_observer():
    reporter = ppe.reporting.Reporter()
    observer = object()
    reporter.add_observer('o', observer)
    with reporter:
        ppe.reporting.report({'x': 1}, observer)
    observation = reporter.observation
    assert 'o/x' in observation
    assert observation['o/x'] == 1


def test_report_with_unregistered_observer():
    reporter = ppe.reporting.Reporter()
    observer = object()
    with reporter:
        with pytest.raises(KeyError):
            ppe.reporting.report({'x': 1}, observer)


def test_report_scope():
    reporter = ppe.reporting.Reporter()
    observation = {}

    with reporter:
        with ppe.reporting.report_scope(observation):
            ppe.reporting.report({'x': 1})

    assert 'x' in observation
    assert observation['x'] == 1
    assert 'x' not in reporter.observation


def test_report_tensor_detached():
    reporter = ppe.reporting.Reporter()
    x = torch.tensor(numpy.array(1, 'float32'), requires_grad=True)
    with reporter:
        ppe.reporting.report({'x': x})
    observation = reporter.observation
    assert 'x' in observation
    assert not observation['x'].requires_grad
    assert x.requires_grad


# ppe.reporting.Summary

def test_summary_basic():
    summary = ppe.reporting.Summary()
    summary.add(torch.Tensor(numpy.array(1, 'float32')))
    summary.add(torch.Tensor(numpy.array(-2, 'float32')))

    mean = summary.compute_mean()
    numpy.testing.assert_allclose(mean.numpy(), numpy.array(-0.5, 'f'))

    mean, std = summary.make_statistics()
    numpy.testing.assert_allclose(mean.numpy(), numpy.array(-0.5, 'f'))
    numpy.testing.assert_allclose(std.numpy(), numpy.array(1.5, 'f'))


def test_summary_int():
    summary = ppe.reporting.Summary()
    summary.add(1)
    summary.add(2)
    summary.add(3)

    mean = summary.compute_mean()
    numpy.testing.assert_allclose(mean, 2)

    mean, std = summary.make_statistics()
    numpy.testing.assert_allclose(mean, 2)
    numpy.testing.assert_allclose(std, numpy.sqrt(2. / 3.))


def test_summary_float():
    summary = ppe.reporting.Summary()
    summary.add(1.)
    summary.add(2.)
    summary.add(3.)

    mean = summary.compute_mean()
    numpy.testing.assert_allclose(mean, 2.)

    mean, std = summary.make_statistics()
    numpy.testing.assert_allclose(mean, 2.)
    numpy.testing.assert_allclose(std, numpy.sqrt(2. / 3.))


def test_summary_weight():
    summary = ppe.reporting.Summary()
    summary.add(1., 0.5)
    summary.add(2., numpy.array(0.4))
    summary.add(3., torch.autograd.Variable(torch.Tensor(numpy.array(0.3))))

    mean = summary.compute_mean()
    val = (1 * 0.5 + 2 * 0.4 + 3 * 0.3) / (0.5 + 0.4 + 0.3)
    numpy.testing.assert_allclose(mean.numpy(), val)


def _nograd(v):
    if isinstance(v, torch.Tensor):
        return v.detach()
    return v


def _check_summary_serialize(value1, value2, value3):
    summary = ppe.reporting.Summary()
    summary.add(value1)
    summary.add(value2)

    summary2 = ppe.reporting.Summary()
    with tempfile.NamedTemporaryFile() as f:
        f.close()
        torch.save(summary.state_dict(), f.name)
        # Load tensors in CPU to simulate a snapshot restore
        summary2.load_state_dict(
            torch.load(f.name, map_location=torch.device('cpu')))
    summary2.add(value3)

    expected_mean = float((value1 + value2 + value3) / 3.)
    expected_std = math.sqrt(
        (value1**2 + value2**2 + value3**2) / 3. - expected_mean**2)

    mean = summary2.compute_mean()
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu()
    numpy.testing.assert_allclose(mean, expected_mean)

    mean, std = summary2.make_statistics()
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu()
    if isinstance(std, torch.Tensor):
        std = std.cpu()
    numpy.testing.assert_allclose(mean, expected_mean)
    numpy.testing.assert_allclose(std, expected_std)


def test_serialize_array_float():
    _check_summary_serialize(
        numpy.array(1.5, numpy.float32),
        numpy.array(2.0, numpy.float32),
        # sum of the above two is non-integer
        numpy.array(3.5, numpy.float32))


def test_serialize_array_int():
    _check_summary_serialize(
        numpy.array(1, numpy.int32),
        numpy.array(-2, numpy.int32),
        numpy.array(2, numpy.int32))


def test_serialize_scalar_float():
    _check_summary_serialize(
        1.5, 2.0,
        # sum of the above two is non-integer
        3.5)


def test_serialize_scalar_int():
    _check_summary_serialize(1, -2, 2)


def test_serialize_tensor():
    _check_summary_serialize(
        torch.tensor(1.5),
        torch.tensor(2.0),
        torch.tensor(3.5))


def test_serialize_tensor_cuda():
    _check_summary_serialize(
        torch.tensor(1.5).cuda(),
        torch.tensor(2.0).cuda(),
        torch.tensor(3.5).cuda())


def test_serialize_tensor_with_grad():
    _check_summary_serialize(
        torch.tensor(1.5, requires_grad=True),
        torch.tensor(2.0, requires_grad=True),
        3.5)


# ppe.reporting.DictSummary

def _check_dict_summary(summary, data):
    mean = summary.compute_mean()
    assert set(mean.keys()) == set(data.keys())
    for name in data.keys():
        m = sum(data[name]) / float(len(data[name]))
        numpy.testing.assert_allclose(mean[name], m)

    stats = summary.make_statistics()
    assert (
        set(stats.keys())
        == set(data.keys()).union(name + '.std' for name in data.keys()))
    for name in data.keys():
        m = sum(data[name]) / float(len(data[name]))
        s = numpy.sqrt(
            sum(x * x for x in data[name]) / float(len(data[name]))
            - m * m)
        numpy.testing.assert_allclose(stats[name], m)
        numpy.testing.assert_allclose(stats[name + '.std'], s)


def test_dict_summary():
    summary = ppe.reporting.DictSummary()
    summary.add({'numpy': numpy.array(3, 'f'), 'int': 1, 'float': 4.})
    summary.add({'numpy': numpy.array(1, 'f'), 'int': 5, 'float': 9.})
    summary.add({'numpy': numpy.array(2, 'f'), 'int': 6, 'float': 5.})
    summary.add({'numpy': numpy.array(3, 'f'), 'int': 5, 'float': 8.})

    _check_dict_summary(summary, {
        'numpy': (3., 1., 2., 3.),
        'int': (1, 5, 6, 5),
        'float': (4., 9., 5., 8.),
    })


def test_dit_summary_sparse():
    summary = ppe.reporting.DictSummary()
    summary.add({'a': 3., 'b': 1.})
    summary.add({'a': 1., 'b': 5., 'c': 9.})
    summary.add({'b': 6.})
    summary.add({'a': 3., 'b': 5., 'c': 8.})

    _check_dict_summary(summary, {
        'a': (3., 1., 3.),
        'b': (1., 5., 6., 5.),
        'c': (9., 8.),
    })


def test_dict_summary_weight():
    summary = ppe.reporting.DictSummary()
    summary.add({'a': (1., 0.5)})
    summary.add({'a': (2., numpy.array(0.4))})
    summary.add(
        {'a': (3., torch.autograd.Variable(torch.Tensor(numpy.array(0.3))))})

    mean = summary.compute_mean()
    val = (1 * 0.5 + 2 * 0.4 + 3 * 0.3) / (0.5 + 0.4 + 0.3)
    numpy.testing.assert_allclose(mean['a'].numpy(), val)

    arr = numpy.array([0.5])
    with pytest.raises(ValueError):
        summary.add({'a': (4., arr)})

    var = torch.autograd.Variable(torch.Tensor(numpy.array([0.5])))
    with pytest.raises(ValueError):
        summary.add({'a': (4., var)})


def test_dict_summary_serialize():
    summary = ppe.reporting.DictSummary()
    summary.add({'numpy': numpy.array(3, 'f'), 'int': 1, 'float': 4.})
    summary.add({'numpy': numpy.array(1, 'f'), 'int': 5, 'float': 9.})
    summary.add({'numpy': numpy.array(2, 'f'), 'int': 6, 'float': 5.})

    summary2 = ppe.reporting.DictSummary()
    summary2.load_state_dict(summary.state_dict())
    summary2.add({'numpy': numpy.array(3, 'f'), 'int': 5, 'float': 8.})

    _check_dict_summary(summary2, {
        'numpy': (3., 1., 2., 3.),
        'int': (1, 5, 6, 5),
        'float': (4., 9., 5., 8.),
    })


@pytest.mark.parametrize('delimiter', ['/', '.'])
@pytest.mark.parametrize(
    # How the state of the summary is transferred.
    'transfer_protocol',
    [
        'direct',  # Use state_dict() and load_state_dict()
        'torch',  # Use torch.save() and torch.load()
    ])
def test_dict_summary_serialize_names_with_delimiter(
        delimiter, transfer_protocol):
    key1 = 'a{d}b'.format(d=delimiter)
    key2 = '{d}a{d}b'.format(d=delimiter)
    key3 = 'a{d}b{d}'.format(d=delimiter)
    summary = ppe.reporting.DictSummary()
    summary.add({key1: 3., key2: 1., key3: 4.})
    summary.add({key1: 1., key2: 5., key3: 9.})
    summary.add({key1: 2., key2: 6., key3: 5.})

    if transfer_protocol == 'direct':
        summary2 = ppe.reporting.DictSummary()
        summary2.load_state_dict(summary.state_dict())
    else:
        assert transfer_protocol == 'torch'
        f = io.BytesIO()
        torch.save(summary, f)
        summary2 = torch.load(io.BytesIO(f.getvalue()))
    summary2.add({key1: 3., key2: 5., key3: 8.})

    _check_dict_summary(summary2, {
        key1: (3., 1., 2., 3.),
        key2: (1., 5., 6., 5.),
        key3: (4., 9., 5., 8.),
    })


def test_serialize_overwrite_different_names():
    summary = ppe.reporting.DictSummary()
    summary.add({'a': 3., 'b': 1.})
    summary.add({'a': 1., 'b': 5.})

    summary2 = ppe.reporting.DictSummary()
    summary2.add({'c': 5.})
    summary2.load_state_dict(summary.state_dict())

    _check_dict_summary(summary2, {
        'a': (3., 1.),
        'b': (1., 5.),
    })


def test_serialize_overwrite_rollback():
    summary = ppe.reporting.DictSummary()
    summary.add({'a': 3., 'b': 1.})
    summary.add({'a': 1., 'b': 5.})

    state = summary.state_dict()
    summary.add({'a': 2., 'b': 6., 'c': 5.})
    summary.add({'a': 3., 'b': 4., 'c': 6.})
    summary.load_state_dict(state)

    summary.add({'a': 3., 'b': 5., 'c': 8.})

    _check_dict_summary(summary, {
        'a': (3., 1., 3.),
        'b': (1., 5., 5.),
        'c': (8.,),
    })
