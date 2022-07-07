import pytorch_pfn_extras
import torch.onnx.symbolic_helper
from typing import Optional

if pytorch_pfn_extras.requires("1.12.0"):
    import torch.onnx._globals


class _TorchOnnxConstants:
    def __init__(self) -> None:
        if pytorch_pfn_extras.requires("1.12.0"):
            from torch.onnx._constants import onnx_default_opset, onnx_main_opset, onnx_stable_opsets, onnx_constant_folding_opsets
            self.default_opset_version = onnx_default_opset
            self.main_opset_version = onnx_main_opset
            self.stable_opset_versions = onnx_stable_opsets
            self.constant_folding_opset_versions = onnx_constant_folding_opsets
        else:
            self.default_opset_version = torch.onnx.symbolic_helper._default_onnx_opset_version
            self.main_opset_version = torch.onnx.symbolic_helper._onnx_main_opset
            self.stable_opset_versions = torch.onnx.symbolic_helper._onnx_stable_opsets
            if pytorch_pfn_extras.requires("1.11.0"):
                self.constant_folding_opset_versions = torch.onnx.symbolic_helper._constant_folding_opset_versions  # type: ignore[attr-defined]
            else:
                self.constant_folding_opset_version = torch.onnx.constant_folding_opset_versions  # type: ignore[attr-defined]


CONSTANTS = _TorchOnnxConstants()


class _TorchOnnxGlobals:
    @property
    def export_opset_version(self) -> int:
        if pytorch_pfn_extras.requires("1.12.0"):
            return torch.onnx._globals.GLOBALS._export_onnx_opset_version  # type: ignore[no-any-return]
        else:
            return torch.onnx.symbolic_helper._export_onnx_opset_version

    @property
    def operator_export_type(self) -> Optional[torch._C._onnx.OperatorExportTypes]:
        if pytorch_pfn_extras.requires("1.12.0"):
            return torch.onnx._globals.operator_export_type  # type: ignore[no-any-return]
        else:
            return torch.onnx.symbolic_helper._operator_export_type

    @property
    def onnx_shape_inference(self) -> bool:
        if pytorch_pfn_extras.requires("1.12.0"):
            return torch.onnx._globals.onnx_shape_inference  # type: ignore[no-any-return]
        else:
            return torch.onnx.symbolic_helper._onnx_shape_inference


GLOBALS = _TorchOnnxGlobals()
