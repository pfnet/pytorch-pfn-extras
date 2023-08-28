import contextlib
import json
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Union

import torch
from pytorch_pfn_extras.writing import Writer


class Tracer:
    @contextlib.contextmanager
    def add_event(self, name: str) -> Generator[None, None, None]:
        raise NotImplementedError("Tracers must implement add_event")

    def clear(self) -> None:
        raise NotImplementedError("Tracers must implement clear")

    def flush(self, filename: str, writer: Writer) -> None:
        raise NotImplementedError("Tracers must implement flush")


class ChromeTracingSaveFunc:
    def __call__(self, target: Dict[str, Any], file_o: Any) -> None:
        log = json.dumps(target, indent=4)
        file_o.write(bytes(log.encode("ascii")))


class ChromeTracer(Tracer):
    def __init__(
        self,
        max_event_count: Optional[int] = None,
        enable: bool = True,
    ):
        self._enable = enable
        self._event_list: List[Dict[str, Union[str, int, float]]] = []
        self._max_event_count = max_event_count or float("inf")
        self._event_count = 0
        self._is_cuda_available = torch.cuda.is_available()

    @contextlib.contextmanager
    def add_event(self, name: str) -> Generator[None, None, None]:
        begin_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            if self._enable and self._event_count < self._max_event_count:
                if self._is_cuda_available:
                    torch.cuda.synchronize()  # Wait for process to complete
                self._event_count += 1
                duration_ns = time.perf_counter_ns() - begin_ns
                self._event_list.append(
                    dict(
                        name=name,
                        cat="",
                        ph="X",
                        ts=begin_ns / 1000,  # nano sec -> micro sec
                        dur=duration_ns / 1000,  # ditto
                        pid=0,
                        tid=0,
                    )
                )

    def flush(self, filename: str, writer: Writer) -> None:
        if not self._enable:
            return
        # TODO(ecastill): try to work on some append mode manipulating the
        # file pointer and with json.dumps?
        savefun = ChromeTracingSaveFunc()
        writer(
            filename,
            "",  # out_dir arg is ignored in the writer, uses the writer attr
            self._event_list,
            savefun=savefun,
        )

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
        self._event_list = []
        self._event_count = 0


_thread_local = threading.local()


def get_tracer(tracer_cls=ChromeTracer) -> Tracer:
    if not hasattr(_thread_local, "_tracer"):
        _thread_local._tracer = tracer_cls(None, True)
    if _thread_local._tracer.__class__ is not tracer_cls:
        raise TypeError("get_tracer called with a different cls")
    return _thread_local._tracer  # type: ignore[no-any-return]


def clear_tracer() -> None:
    get_tracer().clear()
