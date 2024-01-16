from typing import Any, Callable, cast

import torch
import torch.library

# Libraries used to store the ops definitions
library = torch.library.Library("ppe", "DEF")
library_impl = torch.library.Library("ppe", "IMPL", "CompositeExplicitAutograd")
library_autograd_impl = torch.library.Library("ppe", "IMPL", "Autograd")
library_meta_impl = torch.library.Library("ppe", "IMPL", "Meta")


class OpDesc:
    """Metadata to register an op to torch.library.

    Attributes:
        op (callable): code to be executed in the forward/backward of the op.
        meta (callable): function to perform shape inference for forward/backward
            passes.
        signature (str): Arguments and return type of the function
            ``"(Tensor a, Tensor b) -> Tensor[]"``.
    """

    def __init__(
        self,
        op: Callable[..., Any],
        meta: Callable[..., Any],
        signature: str,
    ) -> None:
        self.op = op
        self.meta = meta
        self.signature = signature


def _get_autograd(name: str) -> Callable[..., Any]:
    class RunBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):  # type: ignore[no-untyped-def]
            ctx.save_for_backward(*args)
            op_h = torch._C._dispatch_find_schema_or_throw(
                f"ppe::{name}_fwd", ""
            )
            return torch._C._dispatch_call_boxed(op_h, *args, **kwargs)

        @staticmethod
        def backward(ctx, *args):  # type: ignore[no-untyped-def]
            i_args = tuple(ctx.saved_tensors)
            op_h = torch._C._dispatch_find_schema_or_throw(
                f"ppe::{name}_bwd", ""
            )
            return torch._C._dispatch_call_boxed(op_h, *(args + i_args), **{})

    return cast(Callable[..., Any], RunBackward.apply)


def register(
    name: str,
    fwd_op: OpDesc,
    bwd_op: OpDesc,
) -> None:
    """
    Register a custom op under ``torch.ops.ppe.name``

    The function appears as a primitive op in the forward and backward
    ``torch.fx.Graph``s after compiling torch code with `aot_autograd` backend.
    Note that for backward functions, all the arguments of the backward pass
    together with the forward arguments are passed to it. This means if forward had
    ``fwd_op(x, y)`` ``x,y`` arguments, the custom bwd_op needs to have a
    signature like``bwd_op(grad_output, x, y)``

    Arguments:
        name (str): name of the op, shows how it is registered in ``torch.ops.ppe``.
        fwd_op (ppe.ops.OpDesc): code that is executed in the forward pass
        bwd_op (ppe.ops.OpDesc): code that is executed in the backward pass
    """
    function_sig = f"{name}{fwd_op.signature}"
    function_fwd_sig = f"{name}_fwd{fwd_op.signature}"
    function_bwd_sig = f"{name}_bwd{bwd_op.signature}"
    for s in (function_sig, function_fwd_sig, function_bwd_sig):
        library.define(s)

    def function(*args):  # type: ignore[no-untyped-def]
        op_h = torch._C._dispatch_find_schema_or_throw(f"ppe::{name}_fwd", "")
        return torch._C._dispatch_call_boxed(op_h, *args, **{})

    library_impl.impl(name, function)
    library_impl.impl(f"{name}_fwd", fwd_op.op)
    library_impl.impl(f"{name}_bwd", bwd_op.op)
    library_meta_impl.impl(f"{name}_fwd", fwd_op.meta)
    library_meta_impl.impl(f"{name}_bwd", bwd_op.meta)
    library_autograd_impl.impl(name, _get_autograd(name))
