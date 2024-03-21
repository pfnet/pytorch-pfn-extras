import io
import multiprocessing
import os
import shutil
import sys
import threading
import types
from typing import (
    IO,
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import torch

_TargetType = Union[Sequence[Any], Mapping[str, Any]]
_SaveFun = Callable[..., None]
_HookFun = Callable[[], None]
_TaskFun = Callable[..., None]
_Worker = TypeVar("_Worker", threading.Thread, multiprocessing.Process)
_FileSystem = Any


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

    def __init__(self, root: Optional[str] = None) -> None:
        if root is None:
            self._root = os.getcwd()
        else:
            self._root = root

    def get_actual_path(self, path: str) -> str:
        return os.path.join(self.root, path)

    def _wrap_fileobject(
        self,
        file_obj: IO[Any],
        file_path: str,
        *args: Any,
        **kwargs: Any,
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
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        closefd: bool = True,
        opener: Optional[Callable[[str, int], int]] = None,
    ) -> IO[Any]:
        file_path = self.get_actual_path(file_path)
        file_obj = io.open(
            file_path,
            mode,
            buffering,
            encoding,
            errors,
            newline,
            closefd,
            opener,
        )
        return self._wrap_fileobject(
            file_obj,
            file_path,
            mode,
            buffering,
            encoding,
            errors,
            newline,
            closefd,
            opener,
        )

    def list(
        self,
        path_or_prefix: Optional[str] = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        if path_or_prefix is not None:
            path_or_prefix = self.get_actual_path(path_or_prefix)
        if recursive:
            if path_or_prefix is None:
                raise ValueError(
                    "'path_or_prefix' must not be none in recursive mode."
                )
            path_or_prefix = path_or_prefix.rstrip("/")
            # plus 1 to include the trailing slash
            prefix_end_index = len(path_or_prefix) + 1
            yield from self._recursive_list(prefix_end_index, path_or_prefix)
        else:
            for file in os.scandir(path_or_prefix):
                yield file.name

    def _recursive_list(
        self,
        prefix_end_index: int,
        path: str,
    ) -> Iterator[str]:
        path = self.get_actual_path(path)
        for file in os.scandir(path):
            yield file.path[prefix_end_index:]
            if file.is_dir():
                yield from self._recursive_list(prefix_end_index, file.path)

    def stat(self, path: str) -> _PosixFileStat:
        path = self.get_actual_path(path)
        return _PosixFileStat(os.stat(path), path)

    def close(self) -> None:
        pass

    def __enter__(self) -> "_PosixFileSystem":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        pass

    def isdir(self, file_path: str) -> bool:
        path = self.get_actual_path(file_path)
        return os.path.isdir(path)

    def mkdir(
        self,
        file_path: str,
        mode: int = 0o777,
        *args: Any,
        dir_fd: Optional[int] = None,
    ) -> None:
        file_path = self.get_actual_path(file_path)
        return os.mkdir(file_path, mode, *args, dir_fd=dir_fd)

    def makedirs(
        self, file_path: str, mode: int = 0o777, exist_ok: bool = False
    ) -> None:
        file_path = self.get_actual_path(file_path)
        return os.makedirs(file_path, mode, exist_ok)

    def exists(self, file_path: str) -> bool:
        return os.path.exists(file_path)

    def rename(self, src: str, dst: str) -> None:
        try:
            return os.replace(src, dst)
        except OSError:
            print(
                "Destination {} is a directory "
                "but source is not".format(dst),
                file=sys.stderr,
            )
            raise

    def remove(self, file_path: str, recursive: bool = False) -> None:
        file_path = self.get_actual_path(file_path)
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
        fs: _FileSystem = None,
        out_dir: str = "",
    ) -> None:
        self._post_save_hooks: List[_HookFun] = []
        self.fs = fs or _PosixFileSystem()
        self.out_dir = out_dir
        self._initialized = False

    def __call__(
        self,
        filename: str,
        out_dir: str,
        target: _TargetType,
        *,
        savefun: Optional[_SaveFun] = None,
        append: bool = False,
    ) -> None:
        """Does the actual writing to the file.

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
        """Finalizes the writer.

        Calling this method on already-finalized Writer does nothing."""
        pass

    def save(
        self,
        filename: str,
        out_dir: str,
        target: _TargetType,
        savefun: _SaveFun,
        append: bool,
        **savefun_kwargs: Any,
    ) -> None:
        if not self._initialized:
            self.initialize(out_dir)

        dest = os.path.join(out_dir, filename)

        save_dir = os.path.dirname(dest)
        self.fs.makedirs(save_dir, exist_ok=True)
        basename = os.path.basename(dest)

        if append:
            with self.fs.open(dest, "ab") as f:
                # HDFS does not support overwrite
                savefun(target, f, **savefun_kwargs)
        else:
            # Some filesystems are not compatible with temp folders, etc
            # so we rely on raw temp files
            prefix = "tmp_{}".format(basename)
            tmppath = os.path.join(save_dir, prefix)
            make_backup = self.fs.exists(dest)
            with self.fs.open(tmppath, "wb") as f:
                savefun(target, f, **savefun_kwargs)
            if make_backup:
                bak = "{}.bak".format(dest)
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

    def _add_cleanup_hook(self, hook_fun: _HookFun) -> None:
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


class StandardWriter(Writer, Generic[_Worker]):
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
            optional, defaults to ``''``
        kwds: Keyword arguments for the ``savefun``.

    .. seealso::

        - :meth:`pytorch_pfn_extras.training.extensions.snapshot`
    """

    def __init__(
        self,
        savefun: _SaveFun = torch.save,
        fs: _FileSystem = None,
        out_dir: str = "",
        **kwds: Any,
    ) -> None:
        super().__init__(fs=fs, out_dir=out_dir)
        self._savefun = savefun
        self._kwds = kwds
        self._worker: Optional[_Worker] = None
        self._started = False
        self._finalized = False

    def __call__(
        self,
        filename: str,
        out_dir: str,
        target: _TargetType,
        *,
        savefun: Optional[_SaveFun] = None,
        append: bool = False,
    ) -> None:
        assert not self._finalized
        if savefun is None:
            savefun = self._savefun
        if self._started:
            self.finalize()
        self._filename = filename
        self._worker = self.create_worker(
            filename,
            out_dir,
            target,
            savefun=savefun,
            append=append,
            **self._kwds,
        )
        self._worker.start()
        self._started = True
        self._finalized = False

    def create_worker(
        self,
        filename: str,
        out_dir: str,
        target: _TargetType,
        *,
        savefun: Optional[_SaveFun] = None,
        append: bool = False,
        **savefun_kwargs: Any,
    ) -> _Worker:
        """Creates a worker for the snapshot.

        This method creates a thread or a process to take a snapshot. The
        created worker must have :meth:`start` and :meth:`join` methods.
        If the worker has an ``exitcode`` attribute (e.g.,
        ``multiprocessing.Process``), the value will be tested.
        """
        raise NotImplementedError

    def finalize(self) -> None:
        if self._finalized:
            return

        if self._worker is None:
            raise RuntimeError("worker is not created")
        try:
            if self._started and not self._finalized:
                self._worker.join()
                exitcode = getattr(self._worker, "exitcode", 0)
                if exitcode != 0:
                    raise RuntimeError(f"exit code is non-zero: {exitcode}")
        finally:
            self._started = False
            self._finalized = True
