import sys
from typing import Any, IO, List, Optional, Union

from IPython.display import display
from ipywidgets import HTML

from pytorch_pfn_extras.training.extensions.print_report import PrintReport

from pytorch_pfn_extras.training.extensions import log_report \
    as log_report_module
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol


class PrintReportNotebook(PrintReport):

    """An extension to print the accumulated results.

    It is aimed to work on jupyter notebook as replacement of `PrintReport`.
    This extension uses the log accumulated by a :class:`LogReport` extension
    to print specified entries of the log in a human-readable format.

    Args:
        entries (list of str ot None): List of keys of observations to print.
            If `None` is passed, automatically infer keys from reported dict.
        log_report (str or LogReport): Log report to accumulate the
            observations. This is either the name of a LogReport extensions
            registered to the manager, or a LogReport instance to use
            internally.
        out: This is not used, argument is kept to be consistent with
            `PrintReport`.

    """

    def __init__(
            self,
            entries: Optional[List[str]] = None,
            log_report: Union[str, log_report_module.LogReport] = 'LogReport',
            out: IO[Any] = sys.stdout,
    ) -> None:
        super(PrintReportNotebook, self).__init__(
            entries=entries, log_report=log_report, out=out
        )
        self._widget = HTML()

    def initialize(self, manager: ExtensionsManagerProtocol) -> None:
        display(self._widget)
        super(PrintReportNotebook, self).initialize(manager)

    @property
    def widget(self) -> HTML:
        return self._widget

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        log_report = self.get_log_report(manager)
        df = log_report.to_dataframe()
        if self._infer_entries:
            # --- update entries ---
            self._update_entries(log_report)
        self._widget.value = df[self._entries].to_html(index=False, na_rep='')
