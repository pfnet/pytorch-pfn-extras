import multiprocessing as mp
import sys

import pytest

from pytorch_pfn_extras.profiler import TimeSummary, time_summary


def test_report():
    summary = TimeSummary()
    with summary.report("foo"):
        pass
    summary.wait()
    with summary.summary() as s:
        assert "foo" in s.compute_mean()
    summary.close()


def worker(summary):
    with summary.report("foo"):
        pass


@pytest.mark.skipif(
    sys.platform == 'win32',
    reason='Multiprocessing not fully supported on Windows')
def test_report_from_other_process():
    summary = TimeSummary()
    p = mp.Process(target=worker, args=(summary,))
    p.start()
    p.join()
    summary.wait()
    with summary.summary() as s:
        assert "foo" in s.compute_mean()
    summary.close()


def worker1():
    with time_summary.report("foo"):
        pass


@pytest.mark.skipif(
    sys.platform == 'win32',
    reason='Multiprocessing not fully supported on Windows')
def test_global_summary():
    time_summary.initialize()
    p = mp.Process(target=worker1)
    p.start()
    p.join()
    time_summary.wait()
    with time_summary.summary() as s:
        assert "foo" in s.compute_mean()


def test_clear():
    summary = TimeSummary()
    summary._summary.add({"foo": 10})
    summary.wait()
    with summary.summary(clear=True) as s:
        assert s.compute_mean() == {"foo": 10}
    with summary.summary(clear=True) as s:
        assert s.compute_mean() == {}
    summary.close()
