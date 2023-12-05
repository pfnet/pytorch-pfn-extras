from typing import Any, Callable

import torch


def register(
    name: str, fwd_op: Callable[..., Any], bwd_op: Callable[..., Any]
) -> None:
    """
    Register a custom op under ``torch.ops.ppe.name``

    The function appears as a primitive op in the forward and backward
    ``torch.fx.Graph``s after compiling torch code with `aot_autograd` backend.

    Arguments:
        name (str): name of the op, shows how it is registered in ``torch.ops.ppe``.
        fwd_op (callable): code that is executed in the forward pass
        bwd_op (callable): code that is executed in the backward pass
    """
    pass
