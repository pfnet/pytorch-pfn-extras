from typing import Any, KeysView, Optional
import warnings

from pytorch_pfn_extras.writing._writer_base import (
    _TargetType, _SaveFun, _FileSystem
)


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
            savefun: Optional[_SaveFun] = None,
            fs: _FileSystem = None,
            out_dir: str = '',
            stats: Optional[KeysView[str]] = None,
            **kwds: Any
    ) -> None:
        self._writer = None
        try:
            import torch.utils.tensorboard
        except ImportError:
            warnings.warn(
                'tensorboard is unavailable. '
                'TensorBoardWriter will do nothing.')
            return
        self._stats = stats
        self._writer = (
            torch.utils.tensorboard.SummaryWriter(  # type: ignore[no-untyped-call]
                log_dir=out_dir, **kwds))

    def __del__(self) -> None:
        self.finalize()

    def __call__(
            self,
            filename: str,
            out_dir: str,
            target: _TargetType,
            *,
            savefun: Optional[_SaveFun] = None,
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
        if self._writer is None:
            return
        stats_cpu = target
        if isinstance(target, list):
            stats_cpu = target[-1]

        if not isinstance(stats_cpu, dict):
            raise TypeError('target must be dict or list of dicts')
        keys = stats_cpu.keys()
        if self._stats is not None:
            keys = self._stats  # type: ignore[assignment]
        for key in keys:
            value = stats_cpu[key]
            self._writer.add_scalar(  # type: ignore[no-untyped-call]
                key, value, stats_cpu['iteration'])

    def finalize(self) -> None:
        if self._writer is not None:
            self._writer.close()  # type: ignore[no-untyped-call]
            self._writer = None
