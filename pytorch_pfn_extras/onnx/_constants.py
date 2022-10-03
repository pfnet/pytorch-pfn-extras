import pytorch_pfn_extras
import torch.onnx
import torch.onnx.symbolic_helper

if pytorch_pfn_extras.requires("1.12.0"):
    from torch.onnx._constants import *  # NOQA
else:
    onnx_default_opset = torch.onnx.symbolic_helper._default_onnx_opset_version  # type: ignore
    onnx_main_opset = torch.onnx.symbolic_helper._onnx_main_opset  # type: ignore
    onnx_stable_opsets = torch.onnx.symbolic_helper._onnx_stable_opsets  # type: ignore
    onnx_constant_folding_opsets = torch.onnx.symbolic_helper._constant_folding_opset_versions if pytorch_pfn_extras.requires("1.11.0") else torch.onnx.constant_folding_opset_versions  # type: ignore[attr-defined]
