import sys

from IPython.core.display import display
from ipywidgets import HTML

from pytorch_pfn_extras.training.extensions.print_report import PrintReport


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

    def __init__(self, entries=None, log_report='LogReport', out=sys.stdout):
        super(PrintReportNotebook, self).__init__(
            entries=entries, log_report=log_report, out=out
        )
        self._widget = HTML()

    def initialize(self, trainer):
        display(self._widget)

    @property
    def widget(self):
        return self._widget

    def __call__(self, manager):
        log_report = self.get_log_report(manager)
        df = log_report.to_dataframe()
        if self._infer_entries:
            # --- update entries ---
            self._update_entries(log_report)
        self._widget.value = df[self._entries].to_html(index=False, na_rep='')
