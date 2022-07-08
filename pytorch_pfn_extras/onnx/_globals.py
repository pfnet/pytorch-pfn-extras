import pytorch_pfn_extras
import torch
import torch.onnx.symbolic_helper
from typing import Optional


class _InternalGlobalsBeforeTorch1_11:
    @property
    def export_onnx_opset_version(self) -> int:
        return torch.onnx.symbolic_helper._export_onnx_opset_version

    @property
    def operator_export_type(self) -> Optional[torch._C._onnx.OperatorExportTypes]:
        return torch.onnx.symbolic_helper._operator_export_type

    @property
    def training_mode(self) -> Optional[torch._C._onnx.TrainingMode]:
        return torch.onnx.symbolic_helper._training_mode

    @property
    def onnx_shape_inference(self) -> bool:
        return torch.onnx.symbolic_helper._onnx_shape_inference


if pytorch_pfn_extras.requires("1.12.0"):
    import torch.onnx._globals
    GLOBALS = torch.onnx._globals.GLOBALS
else:
    GLOBALS = _InternalGlobalsBeforeTorch1_11()
