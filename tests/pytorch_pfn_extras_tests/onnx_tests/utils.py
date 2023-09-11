import os
from typing import Callable

import onnx
import onnxruntime as ort
import torch
import pytorch_pfn_extras.onnx.pfto_exporter.export as pfto
from pytorch_pfn_extras.onnx import export_testcase
from pytorch_pfn_extras_tests.onnx_tests.test_export_testcase import _get_output_dir


def run_model_test(
    model: Callable,
    args: tuple,
    check_torch_export=True,
    rtol=1e-05,
    atol=1e-08,
    skip_oxrt=False,
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    strict_trace=True,
    mode="eval",
    use_gpu=False,
    seed=5,
    **kwargs,
) -> onnx.ModelProto:
    torch.manual_seed(seed)

    if mode == "train":
        model.train()
    else:
        assert mode == "eval"
        model.eval()

    if use_gpu and torch.cuda.is_available():
        dev = "cuda"
        model.to(dev)
        args = tuple([t.to(dev) if hasattr(t, "to") else t for t in args])

    if operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN:
        skip_oxrt = True

    rng_state = torch.get_rng_state()
    with pfto._force_tracing():
        expected = model(*args)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)
    expected = torch._C._jit_flatten(expected)[0]

    te_model = None
    if check_torch_export:
        torch.set_rng_state(rng_state)
        pt_dir = _get_output_dir("pt", **kwargs)
        export_testcase(
            model,
            args,
            pt_dir,
            use_pfto=False,
            **kwargs,
        )
        te_model = onnx.load(os.path.join(pt_dir, "model.onnx"))

    torch.set_rng_state(rng_state)
    pf_dir = _get_output_dir("pf", **kwargs)
    actual = export_testcase(
        model,
        args,
        pf_dir,
        strict_trace=strict_trace,
        return_output=True,
        use_pfto=True,
        **kwargs,
    )
    if isinstance(actual, torch.Tensor):
        actual = (actual,)
    expected = torch._C._jit_flatten(expected)[0]
    assert len(actual) == len(expected)

    for a, e in zip(actual, expected):
        if isinstance(a, torch.Tensor) and isinstance(e, torch.Tensor):
            assert torch.isclose(a, e, rtol=rtol, atol=atol).all()

    pfto_model = onnx.load(os.path.join(pf_dir, "model.onnx"))
    if te_model is not None:
        assert len(te_model.graph.output) == len(pfto_model.graph.output)
        assert len(te_model.graph.input) == len(pfto_model.graph.input)

    if skip_oxrt:
        return pfto_model

    ort_session = ort.InferenceSession(os.path.join(pf_dir, "model.onnx"))
    input_names = [i.name for i in pfto_model.graph.input]
    actual = ort_session.run(None, {k: v.cpu().numpy() for k, v in zip(input_names, args)})
    for a, e in zip(actual, expected):
        cmp = torch.isclose(torch.tensor(a), e.cpu(), rtol=rtol, atol=atol)
        assert cmp.all(), f"{cmp.logical_not().count_nonzero()} / {cmp.numel()} values failed"

    return pfto_model
