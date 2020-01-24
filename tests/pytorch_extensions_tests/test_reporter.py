import threading
import time

import numpy
import pytest
import torch

import pytorch_extensions as pe


def test_empty_reporter():
    reporter = pe.Reporter()
    assert reporter.observation == {}


def test_enter_exit():
    reporter1 = pe.Reporter()
    reporter2 = pe.Reporter()
    with reporter1:
        assert pe.get_current_reporter() is reporter1
        with reporter2:
            assert pe.get_current_reporter() is reporter2
        assert pe.get_current_reporter() is reporter1


def test_enter_exit_threadsafe():
    # This test ensures reporter.__enter__ correctly stores the reporter
    # in the thread-local storage.

    def thread_func(reporter, record):
        with reporter:
            # Sleep for a tiny moment to cause an overlap of the context
            # managers.
            time.sleep(0.01)
            record.append(pe.get_current_reporter())

    record1 = []  # The current repoter in each thread is stored here.
    record2 = []
    reporter1 = pe.Reporter()
    reporter2 = pe.Reporter()
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
    reporter1 = pe.Reporter()
    reporter2 = pe.Reporter()
    with reporter1:
        observation = {}
        with reporter2.scope(observation):
            assert pe.get_current_reporter() is reporter2
            assert reporter2.observation is observation
        assert pe.get_current_reporter() is reporter1
        assert reporter2.observation is not observation


def test_add_observer():
    reporter = pe.Reporter()
    observer = object()
    reporter.add_observer('o', observer)

    reporter.report({'x': 1}, observer)

    observation = reporter.observation
    assert 'o/x'in observation
    assert observation['o/x'] == 1
    assert 'x'not in observation


def test_add_observers():
    reporter = pe.Reporter()
    observer1 = object()
    reporter.add_observer('o1', observer1)
    observer2 = object()
    reporter.add_observer('o2', observer2)

    reporter.report({'x': 1}, observer1)
    reporter.report({'y': 2}, observer2)

    observation = reporter.observation
    assert 'o1/x' in observation
    assert observation['o1/x'] == 1
    assert 'o2/y'in observation
    assert observation['o2/y'] == 2
    assert 'x' not in observation
    assert 'y' not in observation
    assert 'o1/y' not in observation
    assert 'o2/x' not in observation


def test_report_without_observer():
    reporter = pe.Reporter()
    reporter.report({'x': 1})

    observation = reporter.observation
    assert 'x'in observation
    assert observation['x'] == 1


# pe.report

def test_report_without_reporter():
    observer = object()
    pe.report({'x': 1}, observer)


def test_report():
    reporter = pe.Reporter()
    with reporter:
        pe.report({'x': 1})
    observation = reporter.observation
    assert 'x' in observation
    assert observation['x'] == 1


def test_report_with_observer():
    reporter = pe.Reporter()
    observer = object()
    reporter.add_observer('o', observer)
    with reporter:
        pe.report({'x': 1}, observer)
    observation = reporter.observation
    assert 'o/x' in observation
    assert observation['o/x'] == 1


def test_report_with_unregistered_observer():
    reporter = pe.Reporter()
    observer = object()
    with reporter:
        with pytest.raises(KeyError):
            pe.report({'x': 1}, observer)


def test_report_scope():
    reporter = pe.Reporter()
    observation = {}

    with reporter:
        with pe.report_scope(observation):
            pe.report({'x': 1})

    assert 'x' in observation
    assert observation['x'] == 1
    assert 'x' not in reporter.observation


# pe.Summary

def test_summary_basic():
    summary = pe.Summary()
    summary.add(torch.Tensor(numpy.array(1, 'float32')))
    summary.add(torch.Tensor(numpy.array(-2, 'float32')))

    mean = summary.compute_mean()
    numpy.testing.assert_allclose(mean.numpy(), numpy.array(-0.5, 'f'))

    mean, std = summary.make_statistics()
    numpy.testing.assert_allclose(mean.numpy(), numpy.array(-0.5, 'f'))
    numpy.testing.assert_allclose(std.numpy(), numpy.array(1.5, 'f'))


def test_summary_int():
    summary = pe.Summary()
    summary.add(1)
    summary.add(2)
    summary.add(3)

    mean = summary.compute_mean()
    numpy.testing.assert_allclose(mean, 2)

    mean, std = summary.make_statistics()
    numpy.testing.assert_allclose(mean, 2)
    numpy.testing.assert_allclose(std, numpy.sqrt(2. / 3.))


def test_summary_float():
    summary = pe.Summary()
    summary.add(1.)
    summary.add(2.)
    summary.add(3.)

    mean = summary.compute_mean()
    numpy.testing.assert_allclose(mean, 2.)

    mean, std = summary.make_statistics()
    numpy.testing.assert_allclose(mean, 2.)
    numpy.testing.assert_allclose(std, numpy.sqrt(2. / 3.))


def test_summary_weight():
    summary = pe.Summary()
    summary.add(1., 0.5)
    summary.add(2., numpy.array(0.4))
    summary.add(3., torch.autograd.Variable(torch.Tensor(numpy.array(0.3))))

    mean = summary.compute_mean()
    val = (1 * 0.5 + 2 * 0.4 + 3 * 0.3) / (0.5 + 0.4 + 0.3)
    numpy.testing.assert_allclose(mean.numpy(), val)


# pe.DictSummary

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
    summary = pe.DictSummary()
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
    summary = pe.DictSummary()
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
    summary = pe.DictSummary()
    summary.add({'a': (1., 0.5)})
    summary.add({'a': (2., numpy.array(0.4))})
    summary.add({'a': (3., torch.autograd.Variable(torch.Tensor(numpy.array(0.3))))})

    mean = summary.compute_mean()
    val = (1 * 0.5 + 2 * 0.4 + 3 * 0.3) / (0.5 + 0.4 + 0.3)
    numpy.testing.assert_allclose(mean['a'].numpy(), val)

    arr = numpy.array([0.5])
    with pytest.raises(ValueError):
        summary.add({'a': (4., arr)})

    var = torch.autograd.Variable(torch.Tensor(numpy.array([0.5])))
    with pytest.raises(ValueError):
        summary.add({'a': (4., var)})
