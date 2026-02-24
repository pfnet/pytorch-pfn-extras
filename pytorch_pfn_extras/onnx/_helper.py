import torch
from typing import Callable, Any, Type, TypeVar

import pytorch_pfn_extras as ppe


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


T = TypeVar('T')


# Using hack from https://stackoverflow.com/a/56856290
def suppress_symbolic_warnings(cls: Type[T]) -> Type[T]:
    global torch
    assert issubclass(cls, torch.autograd.Function)
    assert hasattr(cls, "symbolic")

    if (not ppe.requires("1.13")) or ppe.requires("2.0"):
        return cls

    import torch.onnx._internal.jit_utils
    import torch.onnx._globals

    orig_symbolic = cls.symbolic

    # Untyped due to type checker in torch.onnx
    @staticmethod  # type: ignore[misc]
    def new_symbolic(g, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(g, torch._C.Graph):
            ctx = torch.onnx._internal.jit_utils.GraphContext(
                graph=g,
                block=g.block(),
                opset=torch.onnx._globals.GLOBALS.export_onnx_opset_version,
                original_node=None,  # type: ignore[arg-type]
                params_dict=torch.onnx.utils._params_dict,
                env={},
            )
            return orig_symbolic(ctx, *args, **kwargs)
        return orig_symbolic(g, *args, **kwargs)

    cls.symbolic = new_symbolic

    return cls
