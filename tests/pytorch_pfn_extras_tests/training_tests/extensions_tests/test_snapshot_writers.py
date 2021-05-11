import multiprocessing
import threading
import tempfile
from unittest import mock

from pytorch_pfn_extras import writing


spshot_writers_path = 'pytorch_pfn_extras.writing'


def test_simple_writer():
    target = mock.MagicMock()
    w = writing.SimpleWriter()
    w.save = mock.MagicMock()
    with tempfile.TemporaryDirectory() as tempd:
        w('myfile.dat', tempd, target)

    assert w.save.call_count == 1


def test_standard_writer():
    target = mock.MagicMock()
    w = writing.StandardWriter()
    worker = mock.MagicMock()
    name = spshot_writers_path + '.StandardWriter.create_worker'
    with mock.patch(name, return_value=worker):
        with tempfile.TemporaryDirectory() as tempd:
            w('myfile.dat', tempd, target)
            w('myfile.dat', tempd, target)
            w.finalize()

        assert worker.start.call_count == 2
        assert worker.join.call_count == 2


def test_thread_writer_create_worker():
    target = mock.MagicMock()
    w = writing.ThreadWriter()
    with tempfile.TemporaryDirectory() as tempd:
        worker = w.create_worker('myfile.dat', tempd, target, append=False)
        assert isinstance(worker, threading.Thread)
        w('myfile2.dat', tempd, 'test')
        w.finalize()


def test_process_writer_create_worker():
    target = mock.MagicMock()
    w = writing.ProcessWriter()
    with tempfile.TemporaryDirectory() as tempd:
        worker = w.create_worker('myfile.dat', tempd, target, append=False)
        assert isinstance(worker, multiprocessing.Process)
        w('myfile2.dat', tempd, 'test')
        w.finalize()


def test_queue_writer():
    target = mock.MagicMock()
    q = mock.MagicMock()
    consumer = mock.MagicMock()
    names = [spshot_writers_path + '.QueueWriter.create_queue',
             spshot_writers_path + '.QueueWriter.create_consumer']
    with mock.patch(names[0], return_value=q):
        with mock.patch(names[1], return_value=consumer):
            w = writing.QueueWriter()

            with tempfile.TemporaryDirectory() as tempd:
                w('myfile.dat', tempd, target)
                w('myfile.dat', tempd, target)
                w.finalize()

            assert consumer.start.call_count == 1
            assert q.put.call_count == 3
            assert q.join.call_count, 1
            assert consumer.join.call_count == 1


def test_queue_writer_consume():
    names = [spshot_writers_path + '.QueueWriter.create_queue',
             spshot_writers_path + '.QueueWriter.create_consumer']
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
