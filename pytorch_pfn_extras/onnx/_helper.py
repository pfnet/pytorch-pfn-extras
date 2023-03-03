import torch
from typing import Callable, Any


def _detach(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach()
    elif isinstance(x, list):
        return [_detach(elem) for elem in x]
    elif isinstance(x, tuple):
        return tuple([_detach(elem) for elem in x])
    elif isinstance(x, dict):
        return {k: _detach(v) for k, v in x.items()}
    else:
        return x


def no_grad(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    with torch.no_grad():  # type: ignore[no-untyped-call]
        out = fn(*args, **kwargs)
    # torch.no_grad() does not export `detach` op when tracing
    return _detach(out)
