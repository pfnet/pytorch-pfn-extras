import multiprocessing
import io
import os
import queue
import shutil
import sys
import threading
import types
from typing import (
    Any, Callable, Dict, Generator, Generic, IO, KeysView, List,
    Optional, Tuple, Type, TypeVar, Union,
)

import torch


TargetType = Union[List[Any], Dict[str, Any]]
SaveFun = Callable[..., None]
HookFun = Callable[[], None]
TaskFun = Callable[..., None]
Worker = TypeVar('Worker', threading.Thread, multiprocessing.Process)


class _PosixFileStat:
    def __init__(self, _stat: os.stat_result, filename: str) -> None:
        self.filename = filename
        self.last_modified = _stat.st_mtime
        self.last_accessed = _stat.st_atime
        self.created = _stat.st_ctime
        self.mode = _stat.st_mode
        self.size = _stat.st_size


class _PosixFileSystem(object):
    """Class to abstract the calls to the FileSystem

    This class obeys the same interface as PFIO's POSIX
    Filesystems declarations. When using HDFS, PFIO
    handler can be used instead (requires PFIO>1.0).

    This class currently abstracts POSIX
    """
    def __init__(self) -> None:
        pass

    def get_actual_path(self, path: str) -> str:
        return os.path.join(self.root, path)

    def _wrap_fileobject(
            self,
            file_obj: IO[Any],
            file_path: str,
            *args: Any, **kwargs: Any,
    ) -> IO[Any]:
        return file_obj

    @property
    def root(self) -> str:
        return self._root

    @root.setter
    def root(self, root: str) -> None:
        self._root = root

    def open(
            self,
            file_path: str,
            mode: str = 'rb',
            buffering: int = -1,
            encoding: Optional[str] = None,
            errors: Optional[str] = None,
            newline: Optional[str] = None,
            closefd: bool = True,
            opener: Optional[Callable[[str, int], int]] = None,
    ) -> IO[Any]:
        file_obj = io.open(file_path, mode,
                           buffering, encoding, errors,
                           newline, closefd, opener)
        return self._wrap_fileobject(
            file_obj, file_path, mode, buffering, encoding,
            errors, newline, closefd, opener)

    def list(
            self,
            path_or_prefix: Optional[str] = None,
            recursive: bool = False,
    ) -> Generator[str, str, None]:
        if recursive:
            if path_or_prefix is None:
                raise ValueError(
                    "'path_or_prefix' must not be None in recursive mode.")
            path_or_prefix = path_or_prefix.rstrip("/")
            # plus 1 to include the trailing slash
            prefix_end_index = len(path_or_prefix) + 1
            yield from self._recursive_list(prefix_end_index, path_or_prefix)
        else:
            for file in os.scandir(path_or_prefix):
                yield file.name

    def _recursive_list(
            self, prefix_end_index: int, path: str,
    ) -> Generator[str, str, None]:
        for file in os.scandir(path):
            yield file.path[prefix_end_index:]
            if file.is_dir():
                yield from self._recursive_list(prefix_end_index, file.path)

    def stat(self, path: str) -> _PosixFileStat:
        return _PosixFileStat(os.stat(path), path)

    def close(self) -> None:
        pass

    def __enter__(self) -> '_PosixFileSystem':
        return self

    def __exit__(
            self,
            exc_type: Optional[Type[Exception]],
            exc_value: Optional[Exception],
            traceback: Optional[types.TracebackType],
    ) -> None:
        pass

    def isdir(self, file_path: str) -> bool:
        return os.path.isdir(file_path)

    def mkdir(
            self,
            file_path: str,
            mode: int = 0o777,
            *args: Any,
            dir_fd: Optional[int] = None
    ) -> None:
        return os.mkdir(file_path, mode, *args, dir_fd=dir_fd)

    def makedirs(
            self,
            file_path: str,
            mode: int = 0o777,
            exist_ok: bool = False
    ) -> None:
        return os.makedirs(file_path, mode, exist_ok)

    def exists(self, file_path: str) -> bool:
        return os.path.exists(file_path)

    def rename(self, src: str, dst: str) -> None:
        try:
            return os.replace(src, dst)
        except OSError:
            print('Destination {} is a directory '
                  'but source is not'.format(dst),
                  file=sys.stderr)
            raise

    def remove(self, file_path: str, recursive: bool = False) -> None:
        if recursive:
            return shutil.rmtree(file_path)
        if os.path.isdir(file_path):
            return os.rmdir(file_path)

        return os.remove(file_path)


class Writer:

    """Base class of snapshot writers.

    :class:`~pytorch_pfn_extras.training.extensions.Snapshot`
    invokes ``__call__`` of this class every time when taking a snapshot.
    This class determines how the actual saving function will be invoked.

    .. note::
       This extension first writes the serialized object to a temporary file
       and then rename it to the target file name. Thus, if the program stops
       right before the renaming, the temporary file might be left in the
       output directory.

    .. seealso::

        - :meth:`pytorch_pfn_extras.training.extensions.snapshot`
    """

    def __init__(
            self,
            fs: Optional[_PosixFileSystem] = None,
            out_dir: Optional[str] = None,
    ) -> None:
        self._post_save_hooks: List[HookFun] = []
        self.fs = fs or _PosixFileSystem()
        self.out_dir = out_dir
        self._initialized = False

    def __call__(
            self,
            filename: str,
            out_dir: str,
            target: TargetType,
            *,
            savefun: Optional[SaveFun] = None,
            append: bool = False
    ) -> None:
        """Invokes the actual snapshot function.

        This method is invoked by a
        :class:`~pytorch_pfn_extras.training.extensions.Snapshot` object
        every time it takes a snapshot.

        Args:
            filename (str): Name of the file into which the serialized target
                is saved. It is a concrete file name, i.e. not a pre-formatted
                template string.
            out_dir (str): Output directory. Corresponds to
                :py:attr:`ExtensionsManager.out
                 <pytorch_pfn_extras.training.ExtensionsManager.out>`.
            target (dict): Serialized object which will be saved.
            savefun (callable): A callable that accepts a two positional
                arguments (an object to be serialized, file path) like
                `torch.save`.
            append (bool): Mode used to open the file. True to use the append
                mode, False to use the write mode (truncates the file if it
                already exists).
        """
        raise NotImplementedError

    def initialize(self, out_dir: str) -> None:
        self.fs.makedirs(out_dir, exist_ok=True)
        self._initialized = True

    def __del__(self) -> None:
        self.finalize()

    def finalize(self) -> None:
        """Finalizes the writer."""
        pass

    def save(
            self,
            filename: str,
            out_dir: str,
            target: TargetType,
            savefun: SaveFun,
            append: bool,
            **savefun_kwargs: Any,
    ) -> None:
        if self.out_dir is not None:
            out_dir = self.out_dir
        if not self._initialized:
            self.initialize(out_dir)

        dest = os.path.join(out_dir, filename)

        if append:
            with self.fs.open(dest, 'ab') as f:
                # HDFS does not support overwrite
                savefun(target, f, **savefun_kwargs)
        else:
            # Some filesystems are not compatible with temp folders, etc
            # so we rely on raw temp files
            prefix = 'tmp_{}'.format(filename)
            tmppath = os.path.join(out_dir, prefix)
            make_backup = self.fs.exists(dest)
            with self.fs.open(tmppath, 'wb') as f:
                savefun(target, f, **savefun_kwargs)
            if make_backup:
                bak = '{}.bak'.format(dest)
                # Check if another backup file exists
                # due to some unexpected termination of an earlier
                # process
                if self.fs.exists(bak):
                    self.fs.remove(bak)
                self.fs.rename(dest, bak)
            self.fs.rename(tmppath, dest)
            if make_backup:
                self.fs.remove(bak)

        self._post_save()

    def _add_cleanup_hook(self, hook_fun: HookFun) -> None:
        """Adds cleanup hook function.

        Technically, arbitrary user-defined hook can be called, but
        this is intended for cleaning up stale snapshots.

        Args:
            hook_fun (callable): callable function to be called
                right after save is done. It takes no arguments.

        """
        self._post_save_hooks.append(hook_fun)

    def _post_save(self) -> None:
        for hook in self._post_save_hooks:
            hook()


class SimpleWriter(Writer):
    """The most simple snapshot writer.

    This class just passes the arguments to the actual saving function.

    Args:
        savefun: Callable object. It takes three arguments: the output file
            path, the serialized dictionary object, and the optional keyword
            arguments.
        fs: FileSystem abstracting interface to implement all the operations.
            optional, defaults to None
        out_dir: str. Specifies the directory this writer will use.
            It takes precedence over the one specified in `__call__`
            optional, defaults to None
        kwds: Keyword arguments for the ``savefun``.

    .. seealso::

        - :meth:`pytorch_pfn_extras.training.extensions.snapshot`
    """

    def __init__(
            self,
            savefun: SaveFun = torch.save,
            fs: Optional[_PosixFileSystem] = None,
            out_dir: Optional[str] = None,
            **kwds: Any,
    ) -> None:
        super().__init__(fs=fs, out_dir=out_dir)
        self._savefun = savefun
        self._kwds = kwds

    def __call__(
            self,
            filename: str,
            out_dir: str,
            target: TargetType,
            *,
            savefun: Optional[SaveFun] = None,
            append: bool = False
    ) -> None:
        if savefun is None:
            savefun = self._savefun
        self.save(filename, out_dir, target, savefun, append, **self._kwds)


class StandardWriter(Writer, Generic[Worker]):
    """Base class of snapshot writers which use thread or process.

    This class creates a new thread or a process every time when ``__call__``
    is invoked.

    Args:
        savefun: Callable object. It takes three arguments: the output file
            path, the serialized dictionary object, and the optional keyword
            arguments.
        fs: FileSystem abstracting interface to implement all the operations.
            optional, defaults to None
        out_dir: str. Specifies the directory this writer will use.
            It takes precedence over the one specified in `__call__`
            optional, defaults to None
        kwds: Keyword arguments for the ``savefun``.

    .. seealso::

        - :meth:`pytorch_pfn_extras.training.extensions.snapshot`
    """

    def __init__(
            self,
            savefun: SaveFun = torch.save,
            fs: Optional[_PosixFileSystem] = None,
            out_dir: Optional[str] = None,
            **kwds: Any,
    ) -> None:
        super().__init__(fs=fs, out_dir=out_dir)
        self._savefun = savefun
        self._kwds = kwds
        self._worker: Optional[Worker] = None
        self._started = False
        self._finalized = False

    def __call__(
            self,
            filename: str,
            out_dir: str,
            target: TargetType,
            *,
            savefun: Optional[SaveFun] = None,
            append: bool = False
    ) -> None:
        if savefun is None:
            savefun = self._savefun
        if self._started:
            self.finalize()
        self._filename = filename
        self._worker = self.create_worker(
            filename, out_dir, target,
            savefun=savefun, append=append, **self._kwds)
        self._worker.start()
        self._started = True
        self._finalized = False

    def create_worker(
            self,
            filename: str,
            out_dir: str,
            target: TargetType,
            *,
            savefun: Optional[SaveFun] = None,
            append: bool = False,
            **savefun_kwargs: Any,
    ) -> Worker:
        """Creates a worker for the snapshot.

        This method creates a thread or a process to take a snapshot. The
        created worker must have :meth:`start` and :meth:`join` methods.
        If the worker has an ``exitcode`` attribute (e.g.,
        ``multiprocessing.Process``), the value will be tested.
        """
        raise NotImplementedError

    def finalize(self) -> None:
        if self._worker is None:
            raise RuntimeError('worker is not created')
        try:
            if self._started and not self._finalized:
                self._worker.join()
                exitcode = getattr(self._worker, 'exitcode', 0)
                if exitcode != 0:
                    raise RuntimeError(f'exit code is non-zero: {exitcode}')
        finally:
            self._started = False
            self._finalized = True


class ThreadWriter(StandardWriter[threading.Thread]):
    """Snapshot writer that uses a separate thread.

    This class creates a new thread that invokes the actual saving function.

    .. seealso::

        - :meth:`pytorch_pfn_extras.training.extensions.snapshot`
    """

    def __init__(
            self,
            savefun: SaveFun = torch.save,
            fs: Optional[_PosixFileSystem] = None,
            out_dir: Optional[str] = None,
            **kwds: Any
    ) -> None:
        super().__init__(savefun=savefun, fs=fs, out_dir=out_dir, **kwds)

    def _save_with_exitcode(
            self,
            filename: str,
            out_dir: str,
            target: TargetType,
            savefun: SaveFun,
            append: bool,
            **savefun_kwargs: Any,
    ) -> None:
        try:
            self.save(
                filename, out_dir, target, savefun, append, **savefun_kwargs)
        except Exception as e:
            thread = threading.current_thread()
            thread.exitcode = -1  # type: ignore[attr-defined]
            print(
                f'Error: ThreadWriter failed in thread "{thread.name}": '
                f'{type(e).__name__}: {str(e)}', file=sys.stderr)

    def create_worker(
            self,
            filename: str,
            out_dir: str,
            target: TargetType,
            *,
            savefun: Optional[SaveFun] = None,
            append: bool = False,
            **savefun_kwargs: Any,
    ) -> threading.Thread:
        return threading.Thread(
            target=self._save_with_exitcode,
            args=(filename, out_dir, target, savefun, append),
            kwargs=savefun_kwargs)


class ProcessWriter(StandardWriter[multiprocessing.Process]):
    """Snapshot writer that uses a separate process.

    This class creates a new process that invokes the actual saving function.

    .. note::
        Forking a new process from a MPI process might be danger. Consider
        using :class:`ThreadWriter` instead of ``ProcessWriter`` if you are
        using MPI.

    .. seealso::

        - :meth:`pytorch_pfn_extras.training.extensions.snapshot`
    """

    def __init__(
            self,
            savefun: SaveFun = torch.save,
            fs: Optional[_PosixFileSystem] = None,
            out_dir: Optional[str] = None,
            **kwds: Any,
    ) -> None:
        super().__init__(savefun=savefun, fs=fs, out_dir=out_dir, **kwds)

    def create_worker(
            self,
            filename: str,
            out_dir: str,
            target: TargetType,
            *,
            savefun: Optional[SaveFun] = None,
            append: bool = False,
            **savefun_kwargs: Any,
    ) -> multiprocessing.Process:
        return multiprocessing.Process(
            target=self.save,
            args=(filename, out_dir, target, savefun, append),
            kwargs=savefun_kwargs)


QueueType = queue.Queue[Optional[
    Tuple[TaskFun, str, str, TargetType, Optional[SaveFun], bool]]]


class QueueWriter(Writer, Generic[Worker]):
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
            optional, defaults to None
        task: Callable object. Its ``__call__`` must have a same interface to
            ``Writer.__call__``. This object is directly put into the queue.

    .. seealso::

        - :meth:`pytorch_pfn_extras.training.extensions.snapshot`
    """

    def __init__(
            self,
            savefun: SaveFun = torch.save,
            fs: Optional[_PosixFileSystem] = None,
            out_dir: Optional[str] = None,
            task: Optional[TaskFun] = None,
    ) -> None:
        super().__init__(fs=fs, out_dir=out_dir)
        if task is None:
            self._task = self.create_task(savefun)
        else:
            self._task = task
        self._queue = self.create_queue()
        self._consumer: Worker = self.create_consumer(self._queue)
        self._consumer.start()
        self._started = True
        self._finalized = False

    def __call__(
            self,
            filename: str,
            out_dir: str,
            target: TargetType,
            *,
            savefun: Optional[SaveFun] = None,
            append: bool = False
    ) -> None:
        self._queue.put(
            (self._task, filename, out_dir, target, savefun, append))

    def create_task(self, savefun: SaveFun) -> TaskFun:
        return SimpleWriter(savefun=savefun)

    def create_queue(self) -> QueueType:
        raise NotImplementedError

    def create_consumer(self, q: QueueType) -> Worker:
        raise NotImplementedError

    def consume(self, q: QueueType) -> None:
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
            savefun: SaveFun = torch.save,
            fs: Optional[_PosixFileSystem] = None,
            out_dir: Optional[str] = None,
            task: Optional[TaskFun] = None
    ) -> None:
        super().__init__(savefun=savefun, fs=fs, task=task, out_dir=out_dir)

    def create_queue(self) -> QueueType:
        return queue.Queue()

    def create_consumer(self, q: QueueType) -> threading.Thread:
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
            savefun: SaveFun = torch.save,
            fs: Optional[_PosixFileSystem] = None,
            out_dir: Optional[str] = None,
            task: Optional[TaskFun] = None
    ) -> None:
        super().__init__(savefun=savefun, fs=fs, out_dir=out_dir, task=task)

    def create_queue(self) -> QueueType:
        return multiprocessing.JoinableQueue()

    def create_consumer(self, q: QueueType) -> multiprocessing.Process:
        return multiprocessing.Process(target=self.consume, args=(q,))


class TensorBoardWriter(object):
    """ Writer that sends statistics to TensorBoard.

    This class contains a `torch.utils.tensorboard.SummaryWriter`
    object that is used to send the collected statistics to TensorBoard.
    A list of stats can be specified to report only the desired ones.

    Args:
        savefun: Ignored.
        fs: Ignored.
        out_dir: Passed as ``log_dir`` argument to SummaryWriter.
        stats (list): List of statistic keys.
        kwds: Passed as an additional arguments to SummaryWriter.
    """
    def __init__(
            self,
            savefun: Optional[SaveFun] = None,
            fs: Optional[_PosixFileSystem] = None,
            out_dir: Optional[str] = None,
            stats: Optional[KeysView[str]] = None,
            **kwds: Any
    ) -> None:
        import torch.utils.tensorboard
        self._stats = stats
        self._writer = torch.utils.tensorboard.SummaryWriter(  # type: ignore[no-untyped-call] # NOQA: B950
            log_dir=out_dir, **kwds)

    def __del__(self) -> None:
        self.finalize()

    def __call__(
            self,
            filename: str,
            out_dir: str,
            target: TargetType,
            *,
            savefun: Optional[SaveFun] = None,
            append: bool = False,
    ) -> None:
        """Sends the statistics to the TensorBoard.

        Args:
            filename: Ignored.
            out_dir: Ignored.
            target (dict or list): The statistics of the iteration. If given as
                a list, only the last element (assumed to be a dict containing
                the latest iteration statistics) is reported.
            savefun: Ignored.
            append: Ignored.
        """
        stats_cpu = target
        if isinstance(target, list):
            stats_cpu = target[-1]

        if not isinstance(stats_cpu, dict):
            raise TypeError('target must be dict or list of dicts')
        keys = stats_cpu.keys()
        if self._stats is not None:
            keys = self._stats
        for key in keys:
            value = stats_cpu[key]
            self._writer.add_scalar(  # type: ignore[no-untyped-call]
                key, value, stats_cpu['iteration'])

    def finalize(self) -> None:
        self._writer.close()  # type: ignore[no-untyped-call]
