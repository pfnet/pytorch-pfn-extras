import atexit
from contextlib import contextmanager
import os
import time
from typing import Callable, Dict, Generator, Optional, Tuple
import threading
import queue
import multiprocessing as mp
import torch
import weakref

from pytorch_pfn_extras.reporting import DictSummary


Events = Tuple[torch.cuda.Event, torch.cuda.Event]


class _ReportNotification:
    def __init__(
            self,
            summary: 'TimeSummary',
            tag: str,
            use_cuda: bool,
            begin_event: Optional[torch.cuda.Event],
            begin: float,
    ) -> None:
        self._is_completed = True
        self._summary = summary
        self._tag = tag
        self._use_cuda = use_cuda
        self._begin_event = begin_event
        self._begin = begin

    def defer(self) -> None:
        self._is_completed = False

    def complete(self) -> None:
        self._summary.complete_report(
            self._tag, self._use_cuda, self._begin_event, self._begin)


class _CPUWorker:
    def __init__(
            self,
            add: Callable[[str, float], None],
            max_queue_size: int,
    ) -> None:
        self._add = add
        self._max_queue_size = max_queue_size
        self._initialized = False
        self._queue: Optional[mp.JoinableQueue[Optional[Tuple[str, float]]]] = None
        self._thread: Optional[threading.Thread] = None
        self._thread_exited = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self._queue = mp.JoinableQueue(self._max_queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._initialized = True
        self._thread_exited = False

    def finalize(self) -> None:
        if not self._initialized:
            return
        assert self._queue is not None
        assert self._thread is not None
        # In some situations, (when this runs in a subprocess), the queue might have
        # been cut in the worker thread before this function is called
        # due to the non-deterministic shutdown process.
        if not self._thread_exited:
            self._queue.put(None)
        self._queue.join()
        self._queue.close()
        self._queue.join_thread()
        self._initialized = False

    def synchronize(self) -> None:
        assert self._queue is not None
        self._queue.join()

    def put(self, name: str, value: float) -> None:
        assert self._queue is not None
        assert not self._thread_exited
        self._queue.put((name, value))

    def _worker(self) -> None:
        assert self._queue is not None
        while True:
            try:
                v = self._queue.get()
            # If this runs in a subprocess, the cleanup may throw an EOF here
            # before the queue cleanup code is executed
            except EOFError:
                self._thread_exited = True
                break
            if v is None:
                self._queue.task_done()
                break
            name, value = v
            self._add(name, value)
            self._queue.task_done()


_QueueElem = Tuple[str, Tuple[torch.cuda.Event, torch.cuda.Event]]


class _CUDAWorker:
    def __init__(
            self,
            add: Callable[[str, float], None],
            max_queue_size: int,
    ) -> None:
        self._add = add
        self._max_queue_size = max_queue_size
        self._initialized = False
        self._thread: Optional[threading.Thread] = None
        self._queue: Optional['queue.Queue[Optional[_QueueElem]]'] = None
        self._event_lock = threading.Lock()
        self._events: Optional['queue.Queue[torch.cuda.Event]'] = None
        self._thread_exited = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self._queue = queue.Queue(self._max_queue_size)
        self._events = queue.Queue(self._max_queue_size * 2)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._initialized = True
        self._thread_exited = False

    def finalize(self) -> None:
        if not self._initialized:
            return
        assert self._queue is not None
        assert self._thread is not None
        if not self._thread_exited:
            self._queue.put(None)
        self._queue.join()
        self._initialized = False

    def synchronize(self) -> None:
        assert self._queue is not None
        self._queue.join()

    def put(
            self,
            name: str,
            events: Tuple[torch.cuda.Event, torch.cuda.Event],
    ) -> None:
        assert self._queue is not None
        assert not self._thread_exited
        self._queue.put((name, events))

    def _worker(self) -> None:
        assert self._queue is not None
        assert self._events is not None
        while True:
            try:
                v = self._queue.get()
            except EOFError:
                self._thread_exited = True
                break
            if v is None:
                self._queue.task_done()
                break
            name, (begin, end) = v
            end.synchronize()  # type: ignore[no-untyped-call]
            t_ms = begin.elapsed_time(end)  # type: ignore[no-untyped-call]
            self._add(name, t_ms / 1000)
            with self._event_lock:
                self._events.put(begin)
                self._events.put(end)
            self._queue.task_done()

    def get_cuda_event(self) -> torch.cuda.Event:
        assert self._initialized
        assert self._events is not None
        with self._event_lock:
            if self._events.empty():
                event = torch.cuda.Event(  # type: ignore[no-untyped-call]
                    enable_timing=True)
                self._events.put(event)
            return self._events.get()


class _Finalizer:
    def __init__(self, ts: 'TimeSummary') -> None:
        self._ts = weakref.ref(ts)

    def __call__(self) -> None:
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

    def __init__(self, *, max_queue_size: int = 1000, auto_init: bool = True) -> None:
        self._summary_lock = threading.Lock()
        self._summary = DictSummary()
        self._additional_stats: Dict[str, float] = {}

        self._cpu_worker = _CPUWorker(self._add_from_worker, max_queue_size)
        self._cuda_worker: Optional[_CUDAWorker] = None
        if torch.cuda.is_available():
            self._cuda_worker = _CUDAWorker(self._add_from_worker, max_queue_size)

        self._initialized = False
        self._master_pid = os.getpid()
        if auto_init:
            self.initialize()
        atexit.register(_Finalizer(self))

    def __del__(self) -> None:
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

    def finalize(self) -> None:
        if not self._initialized:
            return
        self._cpu_worker.finalize()
        if self._cuda_worker is not None:
            self._cuda_worker.finalize()
        self._initialized = False

    def synchronize(self) -> None:
        self.initialize()
        self._cpu_worker.synchronize()
        if self._cuda_worker is not None:
            self._cuda_worker.synchronize()

    def _add_from_worker(self, name: str, value: float) -> None:
        assert self._initialized
        with self._summary_lock:
            self._summary.add({name: value})
            min_value = self._additional_stats.get(f"{name}.min", value)
            self._additional_stats[f"{name}.min"] = min(value, min_value)
            max_value = self._additional_stats.get(f"{name}.max", value)
            self._additional_stats[f"{name}.max"] = max(value, max_value)

    def add(self, name: str, value: float) -> None:
        self._add_from_worker(name, value)

    @contextmanager
    def summary(
            self,
            clear: bool = False,
    ) -> Generator[Tuple[DictSummary, Dict[str, float]], None, None]:
        self.initialize()
        try:
            with self._summary_lock:
                yield self._summary, self._additional_stats
        finally:
            if clear:
                self._summary = DictSummary()
                self._additional_stats = {}

    def complete_report(
            self,
            tag: str,
            use_cuda: bool,
            begin_event: Optional[torch.cuda.Event],
            begin: float,
    ) -> None:
        end = time.time()
        assert self._cpu_worker._queue is not None
        self._cpu_worker._queue.put((tag, end - begin))
        if use_cuda:
            assert self._cuda_worker is not None
            assert self._cuda_worker._queue is not None
            assert begin_event is not None
            end_event = self._cuda_worker.get_cuda_event()
            end_event.record()  # type: ignore[no-untyped-call]
            self._cuda_worker._queue.put(
                (f"{tag}.cuda", (begin_event, end_event)))

    @contextmanager
    def report(
            self,
            tag: str,
            use_cuda: bool = False,
    ) -> Generator[_ReportNotification, None, None]:
        """Context manager to automatically report execution times.

        The start and completion times are obtained automatically,
        the user only needs to provide a tag to identify the value
        in the summary values.

        Args:
            tag (str): A name to identify the section of code being profiled.
            use_cuda (bool): Indicates if GPU time should also be profiled.
        """
        self.initialize()
        begin_event = None
        if use_cuda:
            assert self._cuda_worker is not None
            begin_event = self._cuda_worker.get_cuda_event()
            begin_event.record()  # type: ignore[no-untyped-call]
        try:
            begin = time.time()
            notification = _ReportNotification(
                self, tag, use_cuda, begin_event, begin)
            yield notification
        finally:
            if notification._is_completed:
                self.complete_report(tag, use_cuda, begin_event, begin)


_thread_local = threading.local()


def get_time_summary() -> TimeSummary:
    if not hasattr(_thread_local, 'time_summary'):
        _thread_local.time_summary = TimeSummary(auto_init=False)
    return _thread_local.time_summary  # type: ignore[no-any-return]
