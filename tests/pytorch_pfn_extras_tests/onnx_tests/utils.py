import tempfile
from typing import Callable, List, Optional

import onnx
import onnxruntime as ort
import torch
from pytorch_pfn_extras.onnx.pfto_exporter.export import export as pfto_export
from pytorch_pfn_extras.onnx.pfto_exporter.torch_reconstruct import reconstruct


def run_model_test(
    model: Callable,
    args: tuple,
    check_torch_export=True,
    rtol=1e-05,
    atol=1e-08,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    skip_oxrt=False,
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    strict_trace=True,
    mode="eval",
    use_gpu=False,
    check_reconstruct=True,
    **kwargs,
) -> onnx.ModelProto:
    if mode == "train":
        model.train()
    else:
        assert mode == "eval"
        model.eval()

    dev = "cpu"
    if use_gpu and torch.cuda.is_available():
        dev = "cuda"
        model.to(dev)
        args = tuple([t.to(dev) if hasattr(t, "to") else t for t in args])

    if operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN:
        skip_oxrt = True
    with tempfile.NamedTemporaryFile() as f:
        f.close()
        rng_state = torch.get_rng_state()
        expected = model(*args)
        if not isinstance(expected, tuple):
            expected = (expected,)

        te_model = None
        if check_torch_export:
            torch.set_rng_state(rng_state)
            with tempfile.NamedTemporaryFile() as torch_f:
                torch_f.close()
                torch.onnx.export(
                    model,
                    args,
                    torch_f.name,
                    input_names=input_names,
                    output_names=output_names,
                    **kwargs,
                )
                te_model = onnx.load(torch_f.name)

        if input_names is None:
            input_names = [f"input_{idx}" for idx, _ in enumerate(args)]
        if output_names is None:
            output_names = [f"output_{idx}" for idx, _ in enumerate(expected)]
        torch.set_rng_state(rng_state)
        actual = pfto_export(
            model,
            args,
            f.name,
            input_names=input_names,
            output_names=output_names,
            strict_trace=strict_trace,
            **kwargs,
        )
        if not isinstance(actual, tuple):
            actual = (actual,)
        assert len(actual) == len(expected)

        for a, e in zip(actual, expected):
            if isinstance(a, torch.Tensor) and isinstance(e, torch.Tensor):
                assert torch.isclose(a, e, rtol=rtol, atol=atol).all()

        pfto_model = onnx.load(f.name)
        if te_model is not None:
            assert len(te_model.graph.output) == len(pfto_model.graph.output)
            assert len(te_model.graph.input) == len(pfto_model.graph.input)

        if check_reconstruct:
            pt, pt_params = reconstruct(pfto_model)
            pt_f = torch._C._create_function_from_graph("forward", pt)

            torch.set_rng_state(rng_state)
            pt_res = pt_f(*args, *[p[1].to(dev) for p in pt_params])
            if isinstance(pt_res, torch.Tensor):
                pt_res = pt_res,
            assert len(pt_res) == len(expected)
            for a, e in zip(pt_res, expected):
                cmp = torch.isclose(a.cpu(), e.cpu(), rtol=rtol, atol=atol)
                assert cmp.all(), f"{cmp.logical_not().count_nonzero()} / {cmp.numel()} values failed"

        if skip_oxrt:
            return pfto_model

        ort_session = ort.InferenceSession(f.name)
        actual = ort_session.run(None, {k: v.cpu().numpy() for k, v in zip(input_names, args)})
        for a, e in zip(actual, expected):
            cmp = torch.isclose(torch.tensor(a), e.cpu(), rtol=rtol, atol=atol)
            assert cmp.all(), f"{cmp.logical_not().count_nonzero()} / {cmp.numel()} values failed"

        return pfto_model
