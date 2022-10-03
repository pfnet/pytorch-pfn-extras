from collections import OrderedDict
import json
from typing import Any, Dict, Iterable, List, Optional

from pytorch_pfn_extras import reporting
from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training.extensions import log_report
from pytorch_pfn_extras.training import trigger as trigger_module
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol
from pytorch_pfn_extras.profiler._time_summary import get_time_summary


class ProfileReport(extension.Extension):
    """Writes the profile results to a file.

    Times are reported by using the
    :meth:`pytorch_pfn_extras.profiler.TimeSummary.report` context manager.

    Args:
        store_keys (iterable of strs): Keys of values to write to the profiler
            report file.
        report_keys (iterable of strs): Keys of values that will be reported.
        trigger: Trigger that decides when to aggregate the result and output
            the values. This is distinct from the trigger of this extension
            itself. If it is a tuple in the form ``<int>, 'epoch'`` or
            ``<int>, 'iteration'``, it is passed to :class:`IntervalTrigger`.
        filename (str): Name of the log file under the output directory. It can
            be a format string: the last result dictionary is passed for the
            formatting. For example, users can use '{iteration}' to separate
            the log files for different iterations. If the log name is None, it
            does not output the log to any file.
        append (bool, optionsl): If the file is JSON Lines or YAML, contents
            will be appended instead of rewriting the file every call.
        format (str, optional): accepted values are `'json'`, `'json-lines'`
            and `'yaml'`.
        writer (writer object, optional): must be callable.
            object to dump the log to. If specified, it needs to have a correct
            `savefun` defined. The writer can override the save location in
            the :class:`pytorch_pfn_extras.training.ExtensionsManager` object
    Args:
        entries (list): list of str

    Returns:
        header (str): header string
        templates (str): template string for print values.
    """
    def __init__(
            self,
            store_keys: Optional[Iterable[str]] = None,
            report_keys: Optional[Iterable[str]] = None,
            trigger: trigger_module.TriggerLike = (1, "epoch"),
            filename: Optional[str] = None,
            append: bool = False,
            format: Optional[str] = None,
            **kwargs: Any,
    ):
        self.time_summary = get_time_summary()
        # Initializes global TimeSummary.
        self.time_summary.initialize()

        if store_keys is None:
            self._store_keys = store_keys
        else:
            self._store_keys = list(store_keys) + [
                key + ".std" for key in store_keys
            ]
        self._report_keys = report_keys
        self._trigger = trigger_module.get_trigger(trigger)
        self._log: List[Any] = []

        log_name = kwargs.get("log_name", "log")
        if filename is None:
            filename = log_name
        del log_name  # avoid accidental use
        self._log_name = filename
        self._writer = kwargs.get('writer', None)

        if format is None and filename is not None:
            if filename.endswith('.jsonl'):
                format = 'json-lines'
            elif filename.endswith('.yaml'):
                format = 'yaml'
            else:
                format = 'json'

        self._append = append
        self._format = format

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        if manager.is_before_training or self._trigger(manager):
            with self.time_summary.summary(clear=True) as s:
                st, additional = s
                stats = st.make_statistics()
                stats.update(additional)
            writer = manager.writer if self._writer is None else self._writer
            # report
            if self._report_keys is not None:
                reports = {
                    f"time.{k}": v
                    for k, v in stats.items()
                    if k in self._report_keys
                }
                reporting.report(reports)

            # output the result
            if self._store_keys is not None:
                stats = {
                    k: v for k, v in stats.items() if k in self._store_keys
                }
            stats_cpu = {k: float(v) for k, v in stats.items()}

            stats_cpu["epoch"] = manager.epoch
            stats_cpu["iteration"] = manager.iteration
            stats_cpu["elapsed_time"] = manager.elapsed_time
            # Recreate dict to fix order of logs
            out = OrderedDict(
                [(k, stats_cpu[k]) for k in sorted(stats_cpu.keys())])

            self._log.append(out)

            # write to the log file
            if self._log_name is not None:
                log_name = self._log_name.format(**out)
                assert self._format is not None
                savefun = log_report.LogWriterSaveFunc(
                    self._format, self._append)
                writer(log_name, out, self._log,  # type: ignore
                       savefun=savefun, append=self._append)
                if self._append:
                    self._log = []

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        if hasattr(self._trigger, "state_dict"):
            state["_trigger"] = self._trigger.state_dict()
        state["_log"] = json.dumps(self._log)
        return state

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        if hasattr(self._trigger, "load_state_dict"):
            self._trigger.load_state_dict(to_load["_trigger"])
        self._log = json.loads(to_load["_log"])

    def finalize(self, manager: ExtensionsManagerProtocol) -> None:
        if self._writer is not None:
            self._writer.finalize()
