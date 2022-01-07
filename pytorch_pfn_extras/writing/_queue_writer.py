import multiprocessing
import queue
import threading
from typing import Generic, Optional, Tuple

import torch

from pytorch_pfn_extras.writing._writer_base import (
    Writer, _TargetType, _SaveFun, _TaskFun, _Worker, _FileSystem,
)
from pytorch_pfn_extras.writing._simple_writer import SimpleWriter


_QueUnit = Optional[Tuple[
    _TaskFun, str, str, _TargetType, Optional[_SaveFun], bool]]


class QueueWriter(Writer, Generic[_Worker]):
    """Base class of queue snapshot writers.

    This class is a base class of snapshot writers that use a queue.
    A Queue is created when this class is constructed, and every time when
    ``__call__`` is invoked, a snapshot task is put into the queue.

    Args:
        savefun: Callable object which is passed to the :meth:`create_task`
            if the task is ``None``. It takes three arguments: the output file
            path, the serialized dictionary object, and the optional keyword
            arguments.
        fs: FileSystem abstracting interface to implement all the operations.
            optional, defaults to None
        out_dir: str. Specifies the directory this writer will use.
            It takes precedence over the one specified in `__call__`
            optional, defaults to ``''``
        task: Callable object. Its ``__call__`` must have a same interface to
            ``Writer.__call__``. This object is directly put into the queue.

    .. seealso::

        - :meth:`pytorch_pfn_extras.training.extensions.snapshot`
    """

    def __init__(
            self,
            savefun: _SaveFun = torch.save,
            fs: _FileSystem = None,
            out_dir: str = '',
            task: Optional[_TaskFun] = None,
    ) -> None:
        super().__init__(fs=fs, out_dir=out_dir)
        self._started = False
        self._finalized = False
        if task is None:
            self._task = self.create_task(savefun)
        else:
            self._task = task
        self._queue = self.create_queue()
        self._consumer: _Worker = self.create_consumer(self._queue)
        self._consumer.start()
        self._started = True

    def __call__(
            self,
            filename: str,
            out_dir: str,
            target: _TargetType,
            *,
            savefun: Optional[_SaveFun] = None,
            append: bool = False
    ) -> None:
        assert not self._finalized
        self._queue.put(
            (self._task, filename, out_dir, target, savefun, append))

    def create_task(self, savefun: _SaveFun) -> _TaskFun:
        return SimpleWriter(savefun=savefun)

    def create_queue(self) -> 'queue.Queue[_QueUnit]':
        raise NotImplementedError

    def create_consumer(self, q: 'queue.Queue[_QueUnit]') -> _Worker:
        raise NotImplementedError

    def consume(self, q: 'queue.Queue[_QueUnit]') -> None:
        while True:
            task = q.get()
            if task is None:
                q.task_done()
                return
            else:
                task[0](
                    task[1], task[2], task[3], savefun=task[4], append=task[5])
                q.task_done()

    def finalize(self) -> None:
        if self._started:
            if not self._finalized:
                self._queue.put(None)
                self._queue.join()
                self._consumer.join()
            self._started = False
        self._finalized = True


class ThreadQueueWriter(QueueWriter[threading.Thread]):
    """Snapshot writer that uses a thread queue.

    This class creates a thread and a queue by :mod:`threading` and
    :mod:`queue` modules
    respectively. The thread will be a consumer of the queue, and the main
    thread will be a producer of the queue.

    .. seealso::

        - :meth:`pytorch_pfn_extras.training.extensions.snapshot`
    """

    def __init__(
            self,
            savefun: _SaveFun = torch.save,
            fs: _FileSystem = None,
            out_dir: str = '',
            task: Optional[_TaskFun] = None
    ) -> None:
        super().__init__(savefun=savefun, fs=fs, task=task, out_dir=out_dir)

    def create_queue(self) -> 'queue.Queue[_QueUnit]':
        return queue.Queue()

    def create_consumer(self, q: 'queue.Queue[_QueUnit]') -> threading.Thread:
        return threading.Thread(target=self.consume, args=(q,))


class ProcessQueueWriter(QueueWriter[multiprocessing.Process]):
    """Snapshot writer that uses process queue.

    This class creates a process and a queue by :mod:`multiprocessing` module.
    The process will be a consumer of this queue, and the main process will be
    a producer of this queue.

    .. note::
        Forking a new process from MPI process might be danger. Consider using
        :class:`ThreadQueueWriter` instead of ``ProcessQueueWriter`` if you are
        using MPI.

    .. seealso::

        - :meth:`pytorch_pfn_extras.training.extensions.snapshot`
    """

    def __init__(
            self,
            savefun: _SaveFun = torch.save,
            fs: _FileSystem = None,
            out_dir: str = '',
            task: Optional[_TaskFun] = None
    ) -> None:
        super().__init__(savefun=savefun, fs=fs, out_dir=out_dir, task=task)

    def create_queue(self) -> 'queue.Queue[_QueUnit]':
        return multiprocessing.JoinableQueue()

    def create_consumer(self, q: 'queue.Queue[_QueUnit]') -> multiprocessing.Process:
        return multiprocessing.Process(target=self.consume, args=(q,))
