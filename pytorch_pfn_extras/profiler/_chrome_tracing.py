import json
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from pytorch_pfn_extras.writing import Writer


class ChromeTracingEvent:
    def __init__(
        self,
        emitter: Any,
        name: str,
        category_list: Optional[List[str]],
        is_cuda_available: bool,
    ):
        self.emitter = emitter
        self.name = name
        self.category_list = category_list or []
        self._is_cuda_available = is_cuda_available

    def __enter__(self) -> None:
        self.begin_ns = time.perf_counter_ns()

    def __exit__(
        self, exc_type: Any, exc_value: Any, tracebac: Any
    ) -> Literal[False]:
        if self._is_cuda_available:
            torch.cuda.synchronize()  # Wait for process to complete

        duration_ns = time.perf_counter_ns() - self.begin_ns
        self.emitter.emit(
            dict(
                name=self.name,
                cat=",".join(self.category_list),
                ph="X",
                ts=self.begin_ns / 1000,  # nano sec -> micro sec
                dur=duration_ns / 1000,  # ditto
                pid=0,
                tid=0,
            )
        )
        return False


class ChromeTracingEventDisabled:
    def __enter__(self) -> None:
        pass

    def __exit__(
        self, exc_type: Any, exc_value: Any, tracebac: Any
    ) -> Literal[False]:
        return False


class ChromeTracingSaveFunc:
    def __call__(self, target: Dict[str, Any], file_o: Any) -> None:
        log = json.dumps(target, indent=4)
        file_o.write(bytes(log.encode("ascii")))


class ChromeTracingEmitter:
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

    def add_event(
        self, name: str, category_list: Optional[List[str]] = None
    ) -> Union[ChromeTracingEvent, ChromeTracingEventDisabled]:
        if not self._enable or self._event_count >= self._max_event_count:
            return ChromeTracingEventDisabled()
        self._event_count += 1
        return ChromeTracingEvent(
            self,
            name=name,
            category_list=category_list,
            is_cuda_available=self._is_cuda_available,
        )

    def emit(self, event: Dict[str, Union[str, int, float]]) -> None:
        if self._enable:
            self._event_list.append(event)

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


def get_chrome_tracer() -> ChromeTracingEmitter:
    if not hasattr(_thread_local, "chrome_tracer"):
        _thread_local.chrome_tracer = ChromeTracingEmitter(None, True)
    return _thread_local.chrome_tracer  # type: ignore[no-any-return]


def clear_chrome_tracer() -> None:
    get_chrome_tracer().clear()
