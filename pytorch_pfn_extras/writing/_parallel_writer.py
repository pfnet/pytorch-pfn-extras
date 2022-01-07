import multiprocessing
import threading
import sys
from typing import Any, Optional

import torch

from pytorch_pfn_extras.writing._writer_base import (
    StandardWriter, _TargetType, _SaveFun, _FileSystem,
)


class ThreadWriter(StandardWriter[threading.Thread]):
    """Snapshot writer that uses a separate thread.

    This class creates a new thread that invokes the actual saving function.

    .. seealso::

        - :meth:`pytorch_pfn_extras.training.extensions.snapshot`
    """

    def __init__(
            self,
            savefun: _SaveFun = torch.save,
            fs: _FileSystem = None,
            out_dir: str = '',
            **kwds: Any
    ) -> None:
        super().__init__(savefun=savefun, fs=fs, out_dir=out_dir, **kwds)

    def _save_with_exitcode(
            self,
            filename: str,
            out_dir: str,
            target: _TargetType,
            savefun: _SaveFun,
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
            target: _TargetType,
            *,
            savefun: Optional[_SaveFun] = None,
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
            savefun: _SaveFun = torch.save,
            fs: _FileSystem = None,
            out_dir: str = '',
            **kwds: Any,
    ) -> None:
        super().__init__(savefun=savefun, fs=fs, out_dir=out_dir, **kwds)

    def create_worker(
            self,
            filename: str,
            out_dir: str,
            target: _TargetType,
            *,
            savefun: Optional[_SaveFun] = None,
            append: bool = False,
            **savefun_kwargs: Any,
    ) -> multiprocessing.Process:
        return multiprocessing.Process(
            target=self.save,
            args=(filename, out_dir, target, savefun, append),
            kwargs=savefun_kwargs)
