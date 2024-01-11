import contextlib
import json
import os
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Type, Union, cast

import torch.cuda
import torch.utils.data
from pytorch_pfn_extras.profiler import _util
from pytorch_pfn_extras.writing import Writer


class Tracer:
    def initialize_writer(self, filename: str, writer: Writer) -> None:
        raise NotImplementedError("Tracers must implement initialize")

    @contextlib.contextmanager
    def add_event(self, name: str) -> Generator[None, None, None]:
        raise NotImplementedError("Tracers must implement add_event")

    def add_remote_event(self, name: str, value: Any) -> None:
        raise NotImplementedError("Tracers must implement add_remote_event")

    def clear(self) -> None:
        raise NotImplementedError("Tracers must implement clear")

    def flush(self, filename: str, writer: Writer) -> None:
        raise NotImplementedError("Tracers must implement flush")

    def enable(self, enable_flag: bool) -> None:
        raise NotImplementedError("Tracers must implement enable")

    def finalize(self) -> None:
        raise NotImplementedError("Tracers must implement finalize")

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("Tracers must implement state_dict")

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        raise NotImplementedError("Tracers must implement load_state_dict")


class DummyTracer(Tracer):
    def initialize_writer(self, filename: str, writer: Writer) -> None:
        pass

    @contextlib.contextmanager
    def add_event(self, name: str) -> Generator[None, None, None]:
        yield

    def clear(self) -> None:
        pass

    def flush(self, filename: str, writer: Writer) -> None:
        pass


class ChromeTracingSaveFunc:
    def _write(self, target: List[Dict[str, Any]], file_o: Any) -> None:
        log = json.dumps(target, indent=4)
        file_o.write(log.encode("ascii"))

    def _append(self, target: List[Dict[str, Any]], file_o: Any) -> None:
        log = "".join(f"\n{json.dumps(o, indent=4)}," for o in target)
        file_o.write(log.encode("ascii"))

    def __call__(
        self,
        target: List[Dict[str, Any]],
        file_o: Any,
        append_mode: bool,
    ) -> None:
        if append_mode:
            self._append(target, file_o)
        else:
            self._write(target, file_o)

    def init(self, target: List[Dict[str, Any]], file_o: Any) -> None:
        log = "["
        file_o.write(log.encode("ascii"))


def load_chrome_trace_as_json(filename: str) -> List[Dict[str, Any]]:
    with open(filename) as f:
        s = f.read()
    if s[-1] != "]":
        s = s[:-1] + "]"
    return cast(List[Dict[str, Any]], json.loads(s))


class ChromeTracer(Tracer):
    """Tracer object that outputs a timeline in Chrome format.

    Args:
        max_event_count (int): Limit the amount of events that can be traced,
            optional.
        enable (bool): Sets the tracer in active state. Optional,
            defaults to ``True``.
    """

    def __init__(
        self,
        max_event_count: Optional[int] = None,
        enable: bool = True,
        append: bool = True,
    ) -> None:
        self._enable = enable
        self._event_list: List[Dict[str, Union[str, int, float]]] = []
        self._max_event_count = max_event_count or float("inf")
        self._event_count = 0
        # Detect if i am a forked process, in such case I send the event to
        # The parent process
        self._is_cuda_available = torch.cuda.is_available()
        self._tracer_queue: _util.QueueWorker = _util.QueueWorker(
            self.add_remote_event, 1000
        )
        self._tracer_queue.initialize()
        self._append = append
        self._savefun = ChromeTracingSaveFunc()

    @contextlib.contextmanager
    def add_event(self, name: str) -> Generator[None, None, None]:
        if (
            not self._enable
            or not _enabled
            or not getattr(_thread_local, "enable", True)
        ):
            yield
            return

        pid = os.getpid()
        is_forked = pid != _main_pid
        is_cuda_available = self._is_cuda_available and not is_forked
        begin_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            if self._enable and self._event_count < self._max_event_count:
                if is_cuda_available:
                    torch.cuda.synchronize()  # Wait for process to complete
                self._event_count += 1
                duration_ns = time.perf_counter_ns() - begin_ns
                tid = threading.get_native_id()
                # Append is thread safe so this should be fine to execute
                # without a lock, the trace does not require the events
                # to be ordered
                event = dict(
                    name=name,
                    cat="",
                    ph="X",
                    ts=begin_ns / 1000,  # nano sec -> micro sec
                    dur=duration_ns / 1000,  # ditto
                    pid=pid,
                    tid=tid,
                )
                if is_forked:
                    self._tracer_queue.put(name, event)
                else:
                    self._event_list.append(
                        cast(Dict[str, Union[str, int, float]], event)
                    )

    def add_remote_event(
        self, name: str, event: Dict[str, Union[str, int, float]]
    ) -> None:
        self._event_list.append(event)

    def initialize_writer(self, filename: str, writer: Writer) -> None:
        if not self._enable:
            return
        if self._append:
            writer(
                filename,
                "",
                {},
                savefun=self._savefun.init,
                append=False,
            )

    def flush(self, filename: str, writer: Writer) -> None:
        if not self._enable:
            return
        self._tracer_queue.synchronize()
        # TODO(ecastill): try to work on some append mode manipulating the
        # file pointer and with json.dumps?
        writer.save(
            filename,
            "",  # out_dir arg is ignored in the writer, uses the writer attr
            self._event_list,
            savefun=self._savefun,
            append=self._append,
            append_mode=self._append,
        )
        if self._append:
            self._event_list.clear()

    def enable(self, enable_flag: bool) -> None:
        self._enable = enable_flag

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        state["_enable"] = self._enable
        state["_event_list"] = json.dumps(self._event_list)
        state["_max_event_count"] = self._max_event_count
        state["_event_count"] = self._event_count
        return state

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        self._enable = to_load["_enable"]
        self._event_list = json.loads(to_load["_event_list"])
        self._max_event_count = to_load["_max_event_count"]
        self._event_count = to_load["_event_count"]

    def clear(self) -> None:
        self._tracer_queue.synchronize()
        self._event_list = []
        self._event_count = 0

    def finalize(self) -> None:
        self._tracer_queue.synchronize()


_tracer: Optional[Tracer] = None
_main_pid = os.getpid()
_enabled: bool = True
_thread_local = threading.local()


def get_tracer(tracer_cls: Type[Tracer] = ChromeTracer, *params: Any) -> Tracer:
    """Gets the current global tracer.

    Args:
        tracer_cls (type of Tracer): type of tracer to create if the global tracer
            hasn't been initialized
    """
    global _tracer
    if _tracer is None:
        _tracer = tracer_cls(*params)
    if _tracer.__class__ is not tracer_cls:
        raise TypeError("get_tracer called with a different cls")
    return _tracer  # type: ignore[no-any-return]


def clear_tracer() -> None:
    """Resets the status of the global tracer."""
    get_tracer().clear()


def enable_global_trace(enable: bool) -> None:
    """Enable or disable tracing for all the threads.

    Args:
        enable (bool): Enable or disable flag.
    """
    global _enabled
    _enabled = enable


def enable_thread_trace(enable: bool) -> None:
    """Enable or disable tracing for the current thread.

    Args:
        enable (bool): Enable or disable flag.
    """
    _thread_local.enable = enable


class TraceableDataset(torch.utils.data.Dataset):
    """Utility class to trace a Dataset inside the DataLoader worker threads.

    Args:
        dataset (torch.utils.data.Dataset): dataset where __getitem__ will
           be traced.
        tag (str): Tag will be used to name the events.
        tracer (Tracer): Tracer object, optional. If ``None`` it defaults to
           `ppe.profile.get_tracer()`.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        tag: str,
        tracer: Optional[Tracer] = None,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._tag = tag
        self._tracer = tracer if tracer is not None else get_tracer()

    def __len__(self) -> Any:
        return len(self._dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: Any) -> Any:
        with self._tracer.add_event(self._tag):
            return self._dataset.__getitem__(idx)
