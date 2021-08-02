import torch
from typing import Callable, Iterable, Optional
from contextlib import contextmanager
from pytorch_pfn_extras.profiler._time_summary import time_summary


@contextmanager
def record(tag: str, metric: Optional[str] = None, use_cuda: bool = False):
    if metric is None:
        metric = tag

    if use_cuda:
        torch.cuda.nvtx.range_push(tag)
    try:
        with torch.autograd.profiler.record_function(tag):
            with time_summary.report(metric, use_cuda):
                yield
    finally:
        if use_cuda:
            torch.cuda.nvtx.range_pop()


def record_function(tag: str, use_cuda: bool = False) -> Callable:
    def wrapper(f):
        def wrapped(*args, **kwargs):
            with record(tag, use_cuda=use_cuda):
                return f(*args, **kwargs)

        return wrapped

    return wrapper


def record_iterable(
    tag: str,
    iter: Iterable,
    divide_metric: bool = False,
    use_cuda: bool = False,
) -> Iterable:
    def wrapped() -> Iterable:
        for i, x in enumerate(iter):
            name = f"{tag}-{i}"
            metric = name if divide_metric else name
            with record(name, metric, use_cuda=use_cuda):
                yield x

    return wrapped()
