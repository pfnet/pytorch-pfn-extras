import multiprocessing as mp
import threading
from typing import Any, Callable, Optional


class QueueWorker:
    def __init__(
        self,
        add: Callable[[str, Any], None],
        max_queue_size: int,
    ) -> None:
        self._add = add
        self._max_queue_size = max_queue_size
        self._initialized = False
        self._queue: Optional[mp.JoinableQueue] = None
        self._thread: Optional[threading.Thread] = None
        self._thread_exited = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._queue = mp.JoinableQueue(self._max_queue_size)
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

    def put(self, name: str, value: Any) -> None:
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
