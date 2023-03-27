import pytorch_pfn_extras
import torch.onnx
import torch.onnx.symbolic_helper

if pytorch_pfn_extras.requires("1.13.0"):
    from torch.onnx._constants import ONNX_DEFAULT_OPSET, ONNX_CONSTANT_FOLDING_MIN_OPSET, ONNX_MAX_OPSET  # type: ignore[attr-defined]
    onnx_default_opset = ONNX_DEFAULT_OPSET
    onnx_constant_folding_opsets = range(ONNX_CONSTANT_FOLDING_MIN_OPSET, ONNX_MAX_OPSET)
    onnx_main_opset = ONNX_MAX_OPSET
elif pytorch_pfn_extras.requires("1.12.0"):
    from torch.onnx._constants import *  # type: ignore # NOQA
else:
    onnx_default_opset = torch.onnx.symbolic_helper._default_onnx_opset_version  # type: ignore
    onnx_main_opset = torch.onnx.symbolic_helper._onnx_main_opset  # type: ignore
    onnx_stable_opsets = torch.onnx.symbolic_helper._onnx_stable_opsets  # type: ignore
    onnx_constant_folding_opsets = torch.onnx.symbolic_helper._constant_folding_opset_versions if pytorch_pfn_extras.requires("1.11.0") else torch.onnx.constant_folding_opset_versions  # type: ignore[attr-defined]
