import pytorch_pfn_extras
import torch
from typing import Optional
from packaging import version

torch_version = version.parse(torch.__version__.split("+")[0])
if version.parse("2.9.0") <= torch_version:
    from torch.onnx._internal.torchscript_exporter import _globals
else:
    from torch.onnx import _globals
GLOBALS = _globals.GLOBALS
