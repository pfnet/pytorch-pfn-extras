# NOTE: type stub (`__init__.pyi`) must be in sync with these public APIs.

try:
    from pytorch_pfn_extras.onnx.export_testcase import export  # NOQA
    from pytorch_pfn_extras.onnx.export_testcase import export_testcase  # NOQA
    from pytorch_pfn_extras.onnx.export_testcase import is_large_tensor  # NOQA
    from pytorch_pfn_extras.onnx.export_testcase import LARGE_TENSOR_DATA_THRESHOLD  # NOQA
    from pytorch_pfn_extras.onnx.annotate import annotate  # NOQA
    from pytorch_pfn_extras.onnx.annotate import apply_annotation  # NOQA
    from pytorch_pfn_extras.onnx.annotate import scoped_anchor  # NOQA
    from pytorch_pfn_extras.onnx._as_output import as_output  # NOQA
    from pytorch_pfn_extras.onnx._grad import grad  # NOQA
    from pytorch_pfn_extras.onnx.load import load_model  # NOQA
    from pytorch_pfn_extras.onnx._helper import no_grad  # NOQA
    import pytorch_pfn_extras.onnx._lax as lax  # NOQA
    available = True
except ImportError:
    available = False
