import pytorch_pfn_extras
import torch
from typing import Optional


if pytorch_pfn_extras.requires("1.12.0"):
    import torch.onnx._globals
    GLOBALS = torch.onnx._globals.GLOBALS

else:
    import torch.onnx.symbolic_helper as symhel

    class _InternalGlobalsBeforeTorch1_11:
        @property
        def export_onnx_opset_version(self) -> int:
            return symhel._export_onnx_opset_version  # type: ignore

        @property
        def operator_export_type(self) -> Optional[torch._C._onnx.OperatorExportTypes]:
            return symhel._operator_export_type  # type: ignore

        @property
        def training_mode(self) -> Optional[torch._C._onnx.TrainingMode]:
            return symhel._training_mode  # type: ignore

        @property
        def onnx_shape_inference(self) -> bool:
            return symhel._onnx_shape_inference  # type: ignore

    GLOBALS = _InternalGlobalsBeforeTorch1_11()  # type: ignore[assignment]
