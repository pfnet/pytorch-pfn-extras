import pytorch_pfn_extras
from typing import cast, Any, Callable, Tuple, Union

if pytorch_pfn_extras.requires("1.13.0"):
    import torch.onnx._internal.registration as reg
    import torch.onnx.utils

    def is_registered_op(opname: str, domain: str, version: int) -> Any:
        return reg.registry.is_registered_op(f"{domain}::{opname}", version)

    Value = torch._C.Value
    SymbolicFunction = Callable[..., Union[Value, Tuple[Value]]]

    def get_registered_op(opname: str, domain: str, version: int) -> SymbolicFunction:
        group = reg.registry.get_function_group(f"{domain}::{opname}")
        assert group is not None
        ret = group.get(version)
        assert ret is not None
        return cast(Callable, ret)  # type: ignore[redundant-cast]

    def register_op(op_type: str, f: Callable, domain: str, opset_version: int) -> None:
        if len(domain) == 0:
            domain = "aten"
        torch.onnx.utils.register_custom_op_symbolic(
            f"{domain}::{op_type}", f, opset_version)  # type: ignore[no-untyped-call]
else:
    from torch.onnx.symbolic_registry import *  # noqa: F401,F403
