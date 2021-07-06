import atexit
from contextlib import contextmanager
import time
from typing import Tuple
from threading import Lock, Thread
from queue import Queue
import multiprocessing as mp
import torch
from pytorch_pfn_extras.reporting import DictSummary


Events = Tuple[torch.cuda.Event, torch.cuda.Event]


class _CPUWorker(object):
    def __init__(self, add, max_queue_size: int):
        self._add = add
        self._queue = mp.JoinableQueue(max_queue_size)
        self._thread = Thread(target=self._worker)
        self._thread.setDaemon(True)
        self._thread.start()

    def close(self):
        self.wait()
        self._queue.put(None)
        self._thread.join()
        self._queue.close()
        self._queue.join_thread()

    def wait(self):
        self._queue.join()

    def _worker(self):
        while True:
            v = self._queue.get()
            if v is None:
                self._queue.task_done()
                break
            name, value = v
            self._add(name, value)
            self._queue.task_done()


class _CUDAWorker(object):
    def __init__(self, add, max_queue_size: int):
        self._add = add
        self._queue = Queue(max_queue_size)
        self._thread = Thread(target=self._worker)
        self._thread.setDaemon(True)
        self._thread.start()
        self._event_lock = Lock()
        self._events = Queue(max_queue_size)

    def close(self):
        self.wait()
        self._queue.put(None)
        self._thread.join()

    def wait(self):
        self._queue.join()

    def _worker(self):
        while True:
            v = self._queue.get()
            if v is None:
                self._queue.task_done()
                break
            name, value = v
            begin, end = value
            end.synchronize()
            t_ms = begin.elapsed_time(end)
            self._add(name, t_ms / 1000)
            with self._event_lock:
                self._events.put(begin)
                self._events.put(end)
            self._queue.task_done()

    def _get_cuda_event(self) -> torch.cuda.Event:
        with self._event_lock:
            if self._events.empty():
                self._events.put(torch.cuda.Event(enable_timing=True))
            return self._events.get()


class TimeSummary(object):
    def __init__(self, max_queue_size: int = 1000):
        self._summary_lock = Lock()
        self._summary = DictSummary()

        self._cpu_worker = _CPUWorker(self._add, max_queue_size)
        if torch.cuda.is_available():
            self._cuda_worker = _CUDAWorker(self._add, max_queue_size)
        else:
            self._cuda_worker = None

    def close(self):
        self.wait()
        self._cpu_worker.close()
        if self._cuda_worker is not None:
            self._cuda_worker.close()

    def _add(self, name, value):
        with self._summary_lock:
            self._summary.add({name: value})

    def wait(self):
        self._cpu_worker.wait()
        if self._cuda_worker is not None:
            self._cuda_worker.wait()

    @contextmanager
    def summary(self, clear: bool = False):
        try:
            with self._summary_lock:
                yield self._summary
        finally:
            if clear:
                self._summary = DictSummary()

    @contextmanager
    def report(self, tag: str, use_cuda: bool = False) -> None:
        if use_cuda:
            begin_event = self._cuda_worker._get_cuda_event()
            begin_event.record()
        try:
            begin = time.time()
            yield
        finally:
            end = time.time()
            self._cpu_worker._queue.put((tag, end - begin))
            if use_cuda:
                end_event = self._cuda_worker._get_cuda_event()
                end_event.record()
                self._cuda_worker._queue.put(
                    (f"{tag}.cuda", (begin_event, end_event)))


time_summary = TimeSummary()
atexit.register(time_summary.close)
