from typing import Any, Optional

import torch

from pytorch_pfn_extras.writing._writer_base import (
    Writer, _TargetType, _SaveFun, _FileSystem
)


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
            optional, defaults to ``''``
        kwds: Keyword arguments for the ``savefun``.

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
        super().__init__(fs=fs, out_dir=out_dir)
        self._savefun = savefun
        self._kwds = kwds

    def __call__(
            self,
            filename: str,
            out_dir: str,
            target: _TargetType,
            *,
            savefun: Optional[_SaveFun] = None,
            append: bool = False
    ) -> None:
        if savefun is None:
            savefun = self._savefun
        self.save(filename, out_dir, target, savefun, append, **self._kwds)
