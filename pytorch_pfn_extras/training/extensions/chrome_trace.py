from typing import Any, Dict, Optional

from pytorch_pfn_extras.profiler._chrome_tracing import (
    clear_chrome_tracer,
    get_chrome_tracer,
)
from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training import trigger as trigger_module
from pytorch_pfn_extras.training._manager_protocol import (
    ExtensionsManagerProtocol,
)


class ChromeTrace(extension.Extension):
    """Writes the profile timeline to a file.

    Times are reported by using the
    :meth:`pytorch_pfn_extras.profiler.TimeSummary.report` context manager.

    Args:
        trigger: Trigger that decides when to aggregate the result and output
            the values. This is distinct from the trigger of this extension
            itself. If it is a tuple in the form ``<int>, 'epoch'`` or
            ``<int>, 'iteration'``, it is passed to :class:`IntervalTrigger`.
        filename (str): Name of the log file under the output directory. It can
            be a format string: the last result dictionary is passed for the
            formatting. For example, users can use '{iteration}' to separate
            the log files for different iterations. If the log name is None, it
            does not output the log to any file.
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
        trigger: trigger_module.TriggerLike = (1, "epoch"),
        filename: Optional[str] = None,
        **kwargs: Any,
    ):
        self._tracer = get_chrome_tracer()

        self._trigger = trigger_module.get_trigger(trigger)

        self._filename = "chrome_trace.json" if filename is None else filename
        self._writer = kwargs.get("writer", None)

    def _flush_trace(self, manager: ExtensionsManagerProtocol) -> None:
        writer = manager.writer if self._writer is None else self._writer

        # write to the log file
        if self._filename is not None:
            self._tracer.flush(self._filename, writer)

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        if not manager.is_before_training and self._trigger(manager):
            self._flush_trace(manager)

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        if hasattr(self._trigger, "state_dict"):
            state["_trigger"] = self._trigger.state_dict()
        state["_tracer"] = self._tracer.state_dict()
        return state

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        if hasattr(self._trigger, "load_state_dict"):
            self._trigger.load_state_dict(to_load["_trigger"])
        self._tracer.load_state_dict(to_load["_tracer"])

    def finalize(self, manager: ExtensionsManagerProtocol) -> None:
        self._flush_trace(manager)
        if self._writer is not None:
            self._writer.finalize()
        clear_chrome_tracer()
