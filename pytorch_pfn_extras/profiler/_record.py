from contextlib import contextmanager
from typing import Any, Callable, Generator, Iterable, Optional, TypeVar

import torch

from pytorch_pfn_extras.profiler._time_summary import time_summary


@contextmanager
def record(
        tag: str,
        metric: Optional[str] = None,
        use_cuda: bool = False,
) -> Generator[None, None, None]:
    if metric is None:
        metric = tag

    if use_cuda:
        torch.cuda.nvtx.range_push(tag)  # type: ignore[no-untyped-call]
    try:
        with torch.autograd.profiler.record_function(tag):
            with time_summary.report(metric, use_cuda) as ntf:
                yield ntf
    finally:
        if use_cuda:
            torch.cuda.nvtx.range_pop()  # type: ignore[no-untyped-call]


_T = TypeVar('_T')


def record_function(
        tag: str,
        use_cuda: bool = False,
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    def wrapper(f: Callable[..., _T]) -> Callable[..., _T]:
        def wrapped(*args: Any, **kwargs: Any) -> _T:
            with record(tag, use_cuda=use_cuda):
                return f(*args, **kwargs)

        return wrapped

    return wrapper


def record_iterable(
    tag: str,
    iter: Iterable[_T],
    divide_metric: bool = False,
    use_cuda: bool = False,
) -> Iterable[_T]:
    def wrapped() -> Iterable[_T]:
        for i, x in enumerate(iter):
            name = f"{tag}-{i}"
            metric = name if divide_metric else tag
            with record(name, metric, use_cuda=use_cuda):
                yield x

    return wrapped()
