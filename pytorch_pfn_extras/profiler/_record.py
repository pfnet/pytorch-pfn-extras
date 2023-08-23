import inspect
import types
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Optional,
    TypeVar,
)

import torch
from pytorch_pfn_extras.profiler import _chrome_tracing, _time_summary
from pytorch_pfn_extras.runtime import runtime_registry

if TYPE_CHECKING:
    from pytorch_pfn_extras.runtime._runtime import DeviceLike


def _infer_tag_name(frame: Optional[types.FrameType], depth: int) -> str:
    for _ in range(depth):
        assert frame is not None
        frame = frame.f_back
    assert frame is not None
    frame_info = inspect.getframeinfo(frame, context=0)
    return "{}:{}:{}".format(
        inspect.getmodulename(frame_info.filename),
        frame_info.lineno,
        frame_info.function,
    )


class _DummyReportNotification(_time_summary._ReportNotification):
    def __init__(self) -> None:
        pass

    def defer(self) -> None:
        pass

    def complete(self) -> None:
        pass


@contextmanager
def dummy_tracer(tag: Optional[str]) -> Generator[None, None, None]:
    yield None


@contextmanager
def tracer(
    tag: str,
    device: "DeviceLike" = "cpu",
    emit_chrome_trace: bool = False,
) -> Generator[None, None, None]:
    runtime_cls = runtime_registry.get_runtime_class_for_device_spec(device)
    runtime_tracer = runtime_cls.trace

    if emit_chrome_trace:
        chrome_tracer = _chrome_tracing.get_chrome_tracer().add_event
    else:
        chrome_tracer = dummy_tracer  # type: ignore[assignment]

    with runtime_tracer(tag, None), chrome_tracer(tag):
        yield


@contextmanager
def record(
    tag: Optional[str],
    metric: Optional[str] = None,
    use_cuda: bool = False,
    enable: bool = True,
    device: "DeviceLike" = "cpu",
    emit_chrome_trace: bool = False,
) -> Generator[_time_summary._ReportNotification, None, None]:
    if not enable:
        yield _DummyReportNotification()
        return

    if tag is None:
        tag = _infer_tag_name(inspect.currentframe(), depth=2)

    if metric is None:
        metric = tag

    if use_cuda:
        torch.cuda.nvtx.range_push(tag)  # type: ignore[no-untyped-call]
    try:
        with tracer(tag, device, emit_chrome_trace):
            time_summary = _time_summary.get_time_summary()
            with time_summary.report(metric, use_cuda) as ntf:
                yield ntf
    finally:
        if use_cuda:
            torch.cuda.nvtx.range_pop()  # type: ignore[no-untyped-call]


_T = TypeVar("_T")


def record_function(
    tag: Optional[str],
    use_cuda: bool = False,
    enable: bool = True,
    device: "DeviceLike" = "cpu",
    emit_chrome_trace: bool = False,
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    def wrapper(f: Callable[..., _T]) -> Callable[..., _T]:
        def wrapped(*args: Any, **kwargs: Any) -> _T:
            name = tag or f.__name__
            with record(
                name,
                None,
                use_cuda,
                enable,
                device,
                emit_chrome_trace,
            ):
                return f(*args, **kwargs)

        return wrapped

    return wrapper


def record_iterable(
    tag: Optional[str],
    iter: Iterable[_T],
    divide_metric: bool = False,
    use_cuda: bool = False,
    enable: bool = True,
    device: "DeviceLike" = "cpu",
    emit_chrome_trace: bool = False,
) -> Iterable[_T]:
    if tag is None:
        tag = _infer_tag_name(inspect.currentframe(), depth=1)

    def wrapped() -> Iterable[_T]:
        for i, x in enumerate(iter):
            name = f"{tag}-{i}"
            metric = name if divide_metric else tag
            with record(
                name,
                metric,
                use_cuda,
                enable,
                device,
                emit_chrome_trace,
            ):
                yield x

    return wrapped()
