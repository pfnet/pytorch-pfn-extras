from typing import Any, Dict, Optional

from pytorch_pfn_extras.profiler._tracing import Tracer, get_tracer
from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training import trigger as trigger_module
from pytorch_pfn_extras.training._manager_protocol import (
    ExtensionsManagerProtocol,
)


class TimelineTrace(extension.Extension):
    """Writes the profile timeline to a file.

    Times are reported by using the
    :meth:`pytorch_pfn_extras.profiler.TimeSummary.report` context manager.

    Args:
        trigger: Trigger that decides when to output the trace.
            This is distinct from the trigger of this extension
            itself. If it is a tuple in the form ``<int>, 'epoch'`` or
            ``<int>, 'iteration'``, it is passed to :class:`IntervalTrigger`.
        enable: Trigger that enables the tracing.
            Note that since the extensions are executed at the end of an iteration
            the tracer will be enabled from the iteration after
            the trigger is fired. If it is a tuple in the form ``<int>, 'epoch'`` or
            ``<int>, 'iteration'``, it is passed to :class:`IntervalTrigger`.
        disable: Trigger that disables the tracing.
            Note that since the extensions are executed at the end of an iteration
            the tracer will be disabled from the iteration after
            the trigger is fired. If it is a tuple in the form ``<int>, 'epoch'`` or
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
        tracer (tracer object, optional):
            object with the tracing logic.  If not specified, the default
            tracer in the thread local storage with be used.
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
        enable: Optional[trigger_module.TriggerLike] = None,
        disable: Optional[trigger_module.TriggerLike] = None,
        tracer: Optional[Tracer] = None,
        **kwargs: Any,
    ):
        self._tracer = tracer if tracer is not None else get_tracer()
        self._enable = None
        if enable is not None:
            self._enable = trigger_module.get_trigger(enable)

        self._disable = None
        if disable is not None:
            self._disable = trigger_module.get_trigger(disable)

        self._trigger = trigger_module.get_trigger(trigger)

        self._filename = "chrome_trace.json" if filename is None else filename
        self._writer = kwargs.get("writer", None)

    def _flush_trace(self, manager: ExtensionsManagerProtocol) -> None:
        # TODO(kaku) It would be nice to be able to select a mode to
        # synchronize the tracer and then flush it as a strict flush_trace()
        writer = manager.writer if self._writer is None else self._writer

        # write to the log file
        if self._filename is not None:
            self._tracer.flush(self._filename, writer)

    def initialize(self, manager: ExtensionsManagerProtocol) -> None:
        writer = manager.writer if self._writer is None else self._writer
        self._tracer.initialize_writer(self._filename, writer)

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        if self._enable is not None and self._enable(manager):
            self._tracer.enable(True)
        if self._disable is not None and self._disable(manager):
            self._tracer.enable(False)

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
        self._tracer.finalize()
