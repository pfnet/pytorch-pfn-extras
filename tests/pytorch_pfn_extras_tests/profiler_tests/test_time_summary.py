import multiprocessing as mp
import sys
import time

import pytest

from pytorch_pfn_extras.profiler import TimeSummary, time_summary


def test_report():
    summary = TimeSummary()
    with summary.report("foo"):
        pass
    summary.wait()
    with summary.summary() as s:
        assert "foo" in s[0].compute_mean()
        assert "foo.min" in s[1]
        assert "foo.max" in s[1]
    summary.close()


def test_report_async():
    summary = TimeSummary()
    with summary.report("afoo") as notification:
        notification.defer()
    time.sleep(0.5)
    # Explicitly call object completion
    notification.complete()
    summary.wait()
    with summary.summary() as s:
        stats = s[0].compute_mean()
        assert "afoo" in stats
        assert abs(0.5 - stats["afoo"]) < 2e-2
        assert abs(0.5 - s[1]["afoo.min"]) < 2e-2
        assert abs(0.5 - s[1]["afoo.max"]) < 2e-2
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
        assert "foo" in s[0].compute_mean()
        assert "foo.min" in s[1]
        assert "foo.max" in s[1]
    summary.close()


def worker1():
    with time_summary.report("foo"):
        pass


@pytest.mark.skipif(
    sys.platform == 'win32',
    reason='Multiprocessing not fully supported on Windows')
def test_global_summary():
    p = mp.Process(target=worker1)
    p.start()
    p.join()
    time_summary.wait()
    with time_summary.summary() as s:
        assert "foo" in s[0].compute_mean()
        assert "foo.min" in s[1]
        assert "foo.max" in s[1]


def test_clear():
    summary = TimeSummary()
    summary._add("foo", 10)
    summary.wait()
    with summary.summary(clear=True) as s:
        assert s[0].compute_mean() == {"foo": 10}
        assert s[1] == {"foo.min": 10, "foo.max": 10}
    with summary.summary(clear=True) as s:
        assert s[0].compute_mean() == {}
        assert s[1] == {}
    summary.close()
