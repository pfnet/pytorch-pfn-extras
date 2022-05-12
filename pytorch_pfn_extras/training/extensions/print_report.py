from copy import deepcopy
import os
import sys
from typing import Any, Dict, IO, List, Optional, Sequence, Tuple, Union

from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training.extensions import log_report \
    as log_report_module
from pytorch_pfn_extras.training.extensions import util
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol


def create_header_and_templates(
        entries: Sequence[str],
) -> Tuple[str, List[Tuple[str, str, str]]]:
    """Construct header and templates from `entries`

    Args:
        entries (list): list of str

    Returns:
        header (str): header string
        templates (str): template string for print values.
    """
    # format information
    entry_widths = [max(10, len(s)) for s in entries]

    header = '  '.join(('{:%d}' % w for w in entry_widths)).format(
        *entries) + '\n'
    templates = []
    for entry, w in zip(entries, entry_widths):
        templates.append((entry, '{:<%dg}  ' % w, ' ' * (w + 2)))
    return header, templates


def filter_and_sort_entries(
        all_entries: List[str],
        unit: str = 'epoch',
) -> List[str]:
    entries = deepcopy(all_entries)
    # TODO(nakago): sort other entries if necessary

    if 'iteration' in entries:
        # move iteration to head
        entries.pop(entries.index('iteration'))
        if unit == 'iteration':
            entries = ['iteration'] + entries
    if 'epoch' in entries:
        # move epoch to head
        entries.pop(entries.index('epoch'))
        if unit == 'epoch':
            entries = ['epoch'] + entries
    if 'elapsed_time' in entries:
        # move elapsed_time to tail
        entries.pop(entries.index('elapsed_time'))
        entries.append('elapsed_time')
    return entries


class PrintReport(extension.Extension):

    """An extension to print the accumulated results.

    This extension uses the log accumulated by a :class:`LogReport` extension
    to print specified entries of the log in a human-readable format.

    Args:
        entries (list of str or None): List of keys of observations to print.
            If `None` is passed, automatically infer keys from reported dict.
        log_report (str or LogReport): Log report to accumulate the
            observations. This is either the name of a LogReport extensions
            registered to the manager, or a LogReport instance to use
            internally.
        out: Stream to print the bar. Standard output is used by default.

    """

    def __init__(
            self,
            entries: Optional[Sequence[str]] = None,
            log_report: Union[str, log_report_module.LogReport] = 'LogReport',
            out: IO[Any] = sys.stdout,
    ) -> None:
        if entries is None:
            self._infer_entries = True
            entries = []
        else:
            self._infer_entries = False
        self._entries = entries
        self._log_report = log_report
        self.__log_looker: Optional[log_report_module._LogLooker] = None
        self._out = out

        # format information
        header, templates = create_header_and_templates(entries)
        self._header: Optional[str] = header  # printed at the first call
        self._templates = templates
        self._all_entries: List[str] = []

    def get_log_report(
            self,
            manager: ExtensionsManagerProtocol,
    ) -> log_report_module.LogReport:
        log_report = self._log_report
        if isinstance(log_report, str):
            ext = manager.get_extension(log_report)
            if not isinstance(ext, log_report_module.LogReport):
                raise TypeError('`log_report` must be LogReport object')
            return ext
        elif isinstance(log_report, log_report_module.LogReport):
            log_report(manager)  # update the log report
            return log_report
        else:
            raise TypeError('log report has a wrong type %s' %
                            type(log_report))

    @property
    def _log_looker(self) -> log_report_module._LogLooker:
        assert self.__log_looker is not None
        return self.__log_looker

    def initialize(self, manager: ExtensionsManagerProtocol) -> None:
        log_report = self.get_log_report(manager)
        self.__log_looker = log_report._log_buffer.emit_new_looker()

    def _update_entries(self, log_report: log_report_module.LogReport) -> None:
        updated_flag = False
        aggregate_entries = self._log_looker.get()
        for obs in aggregate_entries:
            for entry in obs.keys():
                if entry not in self._all_entries:
                    self._all_entries.append(entry)
                    updated_flag = True

        if updated_flag:
            if hasattr(log_report, '_trigger') and hasattr(log_report._trigger,
                                                           'unit'):
                unit = log_report._trigger.unit  # type: ignore[attr-defined]
            else:
                # Failed to infer `unit`, use epoch as default
                unit = 'epoch'
            entries = filter_and_sort_entries(self._all_entries, unit=unit)
            self._entries = entries
            header, templates = create_header_and_templates(entries)
            self._header = header  # printed at the first call
            self._templates = templates

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        log_report = self.get_log_report(manager)

        if self._infer_entries:
            # --- update entries ---
            self._update_entries(log_report)

        out = self._out

        if self._header:
            out.write(self._header)
            self._header = None

        for line in self._log_looker.get():
            # delete the printed contents from the current cursor
            if os.name == 'nt':
                util.erase_console(0, 0)
            else:
                out.write('\033[J')
            self._print(line)
        self._log_looker.clear()

    def state_dict(self) -> Dict[str, Any]:
        log_report = self._log_report
        if isinstance(log_report, log_report_module.LogReport):
            return {'_log_report': log_report.state_dict()}
        return {}

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        log_report = self._log_report
        if isinstance(log_report, log_report_module.LogReport):
            log_report.load_state_dict(to_load['_log_report'])

    def _print(self, observation: log_report_module.Observation) -> None:
        out = self._out
        for entry, template, empty in self._templates:
            if entry in observation:
                out.write(template.format(observation[entry]))
            else:
                out.write(empty)
        out.write('\n')
        if hasattr(out, 'flush'):
            out.flush()
