import pytorch_pfn_extras
import torch.onnx
import torch.onnx.symbolic_helper

from torch.onnx._constants import ONNX_DEFAULT_OPSET, ONNX_CONSTANT_FOLDING_MIN_OPSET, ONNX_MAX_OPSET  # type: ignore[attr-defined]
onnx_default_opset = ONNX_DEFAULT_OPSET
onnx_constant_folding_opsets = range(ONNX_CONSTANT_FOLDING_MIN_OPSET, ONNX_MAX_OPSET)
onnx_main_opset = ONNX_MAX_OPSET
