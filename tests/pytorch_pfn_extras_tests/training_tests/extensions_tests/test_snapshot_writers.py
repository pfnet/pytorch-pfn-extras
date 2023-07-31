import multiprocessing
import os
import tempfile
import threading
from unittest import mock

import pytest
from pytorch_pfn_extras import writing

spshot_writers_path = "pytorch_pfn_extras.writing"


def test_simple_writer():
    target = mock.MagicMock()
    savefun = mock.MagicMock()
    with tempfile.TemporaryDirectory() as tempd:
        w = writing.SimpleWriter(foo=True, out_dir=tempd)
        filename = "myfile.dat"
        w(filename, target, savefun=savefun)
        assert os.path.exists(os.path.join(tempd, filename))
    assert savefun.call_count == 1
    assert savefun.call_args[0][0] == target
    assert savefun.call_args[1]["foo"] is True


def test_standard_writer():
    target = mock.MagicMock()
    worker = mock.MagicMock()
    worker.exitcode = 0
    name = spshot_writers_path + ".StandardWriter.create_worker"
    with mock.patch(name, return_value=worker):
        with tempfile.TemporaryDirectory() as tempd:
            w = writing.StandardWriter(out_dir=tempd)
            w("myfile.dat", target)
            w("myfile.dat", target)
            w.finalize()

        assert worker.start.call_count == 2
        assert worker.join.call_count == 2


def test_thread_writer_create_worker():
    target = mock.MagicMock()
    with tempfile.TemporaryDirectory() as tempd:
        w = writing.ThreadWriter(out_dir=tempd)
        worker = w.create_worker("myfile.dat", target, append=False)
        assert isinstance(worker, threading.Thread)
        w("myfile2.dat", "test")
        w.finalize()


def test_thread_writer_fail():
    with tempfile.TemporaryDirectory() as tempd:
        w = writing.ThreadWriter(savefun=None, out_dir=tempd)
        w("myfile2.dat", "test")
        with pytest.raises(RuntimeError):
            w.finalize()


def test_process_writer_create_worker():
    target = mock.MagicMock()
    with tempfile.TemporaryDirectory() as tempd:
        w = writing.ProcessWriter(out_dir=tempd)
        worker = w.create_worker("myfile.dat", target, append=False)
        assert isinstance(worker, multiprocessing.Process)
        w("myfile2.dat", "test")
        w.finalize()


def test_process_writer_fail():
    with tempfile.TemporaryDirectory() as tempd:
        w = writing.ProcessWriter(savefun=None, out_dir=tempd)
        w("myfile2.dat", "test")
        with pytest.raises(RuntimeError):
            w.finalize()


def test_queue_writer():
    target = mock.MagicMock()
    q = mock.MagicMock()
    consumer = mock.MagicMock()
    names = [
        spshot_writers_path + ".QueueWriter.create_queue",
        spshot_writers_path + ".QueueWriter.create_consumer",
    ]
    with mock.patch(names[0], return_value=q):
        with mock.patch(names[1], return_value=consumer):
            with tempfile.TemporaryDirectory() as tempd:
                w = writing.QueueWriter(out_dir=tempd)
                w("myfile.dat", target)
                w("myfile.dat", target)
                w.finalize()

            assert consumer.start.call_count == 1
            assert q.put.call_count == 3
            assert q.join.call_count, 1
            assert consumer.join.call_count == 1


def test_queue_writer_consume():
    names = [
        spshot_writers_path + ".QueueWriter.create_queue",
        spshot_writers_path + ".QueueWriter.create_consumer",
    ]
    with mock.patch(names[0]):
        with mock.patch(names[1]):
            task = mock.MagicMock()
            q = mock.MagicMock()
            q.get = mock.MagicMock(side_effect=[task, task, None])
            w = writing.QueueWriter()
            w.consume(q)

            assert q.get.call_count == 3
            assert task[0].call_count == 2
            assert q.task_done.call_count == 3
