import atexit
from contextlib import contextmanager
import os
import time
from typing import Tuple
import threading
import queue
import multiprocessing as mp
import torch
import weakref
from pytorch_pfn_extras.reporting import DictSummary


Events = Tuple[torch.cuda.Event, torch.cuda.Event]


class _CPUWorker:
    def __init__(self, add, max_queue_size: int):
        self._add = add
        self._max_queue_size = max_queue_size
        self._initialized = False
        self._queue = None
        self._thread = None

    def initialize(self):
        if self._initialized:
            return
        self._queue = mp.JoinableQueue(self._max_queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._initialized = True

    def finalize(self):
        if not self._initialized:
            return
        self._queue.put(None)
        self._queue.join()
        self._queue.close()
        self._queue.join_thread()
        self._thread.join()
        self._initialized = False

    def synchronize(self):
        assert self._initialized
        self._queue.join()

    def put(self, name, value):
        assert self._initialized
        self._queue.put((name, value))

    def _worker(self):
        while True:
            v = self._queue.get()
            if v is None:
                self._queue.task_done()
                break
            name, value = v
            self._add(name, value)
            self._queue.task_done()


class _CUDAWorker:
    def __init__(self, add, max_queue_size: int):
        self._add = add
        self._max_queue_size = max_queue_size
        self._initialized = False
        self._thread = None
        self._queue = None
        self._event_lock = threading.Lock()
        self._events = None

    def initialize(self):
        if self._initialized:
            return
        self._queue = queue.Queue(self._max_queue_size)
        self._events = queue.Queue(self._max_queue_size * 2)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._initialized = True

    def finalize(self):
        if not self._initialized:
            return
        self._queue.put(None)
        self._queue.join()
        self._thread.join()
        self._initialized = False

    def synchronize(self):
        assert self._initialized
        self._queue.join()

    def put(self, name, events):
        assert self._initialized
        self._queue.put((name, events))

    def _worker(self):
        while True:
            v = self._queue.get()
            if v is None:
                self._queue.task_done()
                break
            name, (begin, end) = v
            end.synchronize()
            t_ms = begin.elapsed_time(end)
            self._add(name, t_ms / 1000)
            with self._event_lock:
                self._events.put(begin)
                self._events.put(end)
            self._queue.task_done()

    def get_cuda_event(self) -> torch.cuda.Event:
        assert self._initialized
        with self._event_lock:
            if self._events.empty():
                self._events.put(torch.cuda.Event(enable_timing=True))
            return self._events.get()


class _Finalizer:
    def __init__(self, ts):
        self._ts = weakref.ref(ts)

    def __call__(self):
        ts = self._ts()
        if ts:
            ts.finalize()


class TimeSummary:
    """Online summarization of execution times.

    `TimeSummary` computes the average and standard deviation of exeuction
    times in both cpu and gpu devices.

    Args:
        max_queue_size (int): Length limit of the internal queues that keep
            reported time info until they are summarized.
        auto_init (bool): Whether to automatically call `initialize()`
            when the instance is created.
    """

    def __init__(self, *, max_queue_size: int = 1000, auto_init: bool = True):
        self._summary_lock = threading.Lock()
        self._summary = DictSummary()
        self._additional_stats = {}

        self._cpu_worker = _CPUWorker(self._add_from_worker, max_queue_size)
        if torch.cuda.is_available():
            self._cuda_worker = _CUDAWorker(self._add_from_worker, max_queue_size)
        else:
            self._cuda_worker = None

        self._initialized = False
        self._master_pid = os.getpid()
        if auto_init:
            self.initialize()
        atexit.register(_Finalizer(self))

    def __del__(self):
        self.finalize()

    def initialize(self) -> None:
        """Initializes the worker threads for TimeSummary.

        Usually you do not have to call it for yourself.
        However in case you directly use ``ppe.time_summary`` outside of
        :class:`pytorch_pfn_extras.training.extensions.ProfileReport`,
        you have to explicitly call ``initialize()`` in advance.
        """
        if self._initialized:
            return
        if os.getpid() != self._master_pid:
            raise RuntimeError(
                "TimeSummary must be initialized in the same process as the "
                "one created the instance. Please call initialize() in the "
                "main process.")
        self._cpu_worker.initialize()
        if self._cuda_worker is not None:
            self._cuda_worker.initialize()
        self._initialized = True

    def finalize(self):
        if not self._initialized:
            return
        self._cpu_worker.finalize()
        if self._cuda_worker is not None:
            self._cuda_worker.finalize()
        self._initialized = False

    def synchronize(self):
        self.initialize()
        self._cpu_worker.synchronize()
        if self._cuda_worker is not None:
            self._cuda_worker.synchronize()

    def _add_from_worker(self, name, value):
        assert self._initialized
        with self._summary_lock:
            self._summary.add({name: value})
            min_value = self._additional_stats.get(f"{name}.min", value)
            self._additional_stats[f"{name}.min"] = min(value, min_value)
            max_value = self._additional_stats.get(f"{name}.max", value)
            self._additional_stats[f"{name}.max"] = max(value, max_value)

    def add(self, name, value):
        self._add_from_worker(name, value)

    @contextmanager
    def summary(self, clear: bool = False):
        self.initialize()
        try:
            with self._summary_lock:
                yield self._summary, self._additional_stats
        finally:
            if clear:
                self._summary = DictSummary()
                self._additional_stats = {}

    @contextmanager
    def report(self, tag: str, use_cuda: bool = False) -> None:
        """Context manager to automatically report execution times.

        The start and completion times are obtained automatically,
        the user only needs to provide a tag to identify the value
        in the summary values.

        Args:
            tag (str): A name to identify the section of code being profiled.
            use_cuda (bool): Indicates if GPU time should also be profiled.
        """
        self.initialize()
        if use_cuda:
            begin_event = self._cuda_worker.get_cuda_event()
            begin_event.record()
        try:
            begin = time.time()
            yield
        finally:
            end = time.time()
            self._cpu_worker.put(tag, end - begin)
            if use_cuda:
                end_event = self._cuda_worker.get_cuda_event()
                end_event.record()
                self._cuda_worker.put(f"{tag}.cuda", (begin_event, end_event))


time_summary = TimeSummary(auto_init=False)
