import pytorch_pfn_extras
import torch
from typing import Optional


_torch_major, _torch_minor = torch.__version__.split(".")[:2]
if (int(_torch_major), int(_torch_minor)) >= (2, 9):
    from torch.onnx._internal.torchscript_exporter import _globals
else:
    from torch.onnx import _globals
GLOBALS = _globals.GLOBALS
