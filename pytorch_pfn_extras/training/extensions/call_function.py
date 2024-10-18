import types
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from pytorch_pfn_extras.reporting import Value, report
from pytorch_pfn_extras.training._manager_protocol import (
    ExtensionsManagerProtocol,
)
from pytorch_pfn_extras.training.extension import PRIORITY_WRITER, Extension


class CallFunction(Extension):
    def __init__(
        self,
        fn: Callable[..., Dict[str, Value]],
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        report_keys: Optional[Sequence[str]] = None,
        report_prefix: Optional[str] = None,
        run_on_error: bool = False,
        priority: int = PRIORITY_WRITER,
    ) -> None:
        """wrapper extension to call functions during the training loop

        Args:
            fn (Callable[..., Dict[str, Value]]): Function to be called via extension.
            args (Optional[Sequence[Any]], optional): Arguments to be passed to the function. Defaults to None.
            kwargs (Optional[Mapping[str, Any]], optional): Keyword arguments you want to pass to the function. Defaults to None.
            report_keys (Optional[Sequence[str]], optional): The key of the value to be reported among the values contained in the function's return dict. Defaults to None.
            report_prefix (Optional[str], optional): If necessary, the prefix to attach to the function's return value when reporting it. Defaults to None.
            run_on_error (bool, optional): Whether or not want to run when an error occurs during the training loop. Defaults to False.
            priority (int, optional): When this Extension will be executed. Defaults to PRIORITY_WRITER.
        """
        self._fn = fn
        self._args = args or []
        self._kwargs = kwargs or {}
        self._report_keys = set(report_keys) if report_keys else None
        self._report_prefix = report_prefix
        self._run_on_error = run_on_error
        self.priority = priority

    def _call(self) -> None:
        out = self._fn(*self._args, **self._kwargs)
        if self._report_keys:
            out = {k: v for k, v in out.items() if k in self._report_keys}
        if self._report_prefix:
            out = {
                "/".join([self._report_prefix, k]): v for k, v in out.items()
            }

        report(out)

    def __call__(self, manager: ExtensionsManagerProtocol) -> Any:
        self._call()

    def on_error(
        self,
        manager: ExtensionsManagerProtocol,
        exc: Exception,
        tb: types.TracebackType,
    ) -> None:
        if self._run_on_error:
            self._call()
