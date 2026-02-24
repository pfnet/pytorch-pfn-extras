from importlib import import_module
from typing import Protocol, cast

import pytorch_pfn_extras
import torch._C._onnx as _C_onnx


class _OnnxGlobalsProtocol(Protocol):
    # NOTE: Keep this in sync with the latest torch (2.10) ONNX exporter globals.
    export_training: bool
    operator_export_type: _C_onnx.OperatorExportTypes
    onnx_shape_inference: bool

    @property
    def training_mode(self) -> _C_onnx.TrainingMode:
        ...

    @training_mode.setter
    def training_mode(self, training_mode: _C_onnx.TrainingMode) -> None:
        ...

    @property
    def export_onnx_opset_version(self) -> int:
        ...

    @export_onnx_opset_version.setter
    def export_onnx_opset_version(self, value: int) -> None:
        ...

    @property
    def in_onnx_export(self) -> bool:
        ...

    @in_onnx_export.setter
    def in_onnx_export(self, value: bool) -> None:
        ...

    @property
    def autograd_inlining(self) -> bool:
        ...

    @autograd_inlining.setter
    def autograd_inlining(self, value: bool) -> None:
        ...


if pytorch_pfn_extras.requires("2.9"):
    _torch_onnx_globals = import_module(
        "torch.onnx._internal.torchscript_exporter._globals"
    )
else:
    _torch_onnx_globals = import_module("torch.onnx._globals")


GLOBALS = cast(_OnnxGlobalsProtocol, _torch_onnx_globals.GLOBALS)
