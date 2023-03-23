from typing import Generator, List, Optional, Tuple, Any
from contextlib import contextmanager

import torch
import torch.onnx
import threading
from pytorch_pfn_extras.onnx._as_output import as_output


_grad_state = threading.local()


@contextmanager
def init_grad_state() -> Generator[None, None, None]:
    _grad_state.n_grad_call = 0
    try:
        yield
    finally:
        _grad_state.n_grad_call = None


def grad(
    output: torch.Tensor,
    inputs: Tuple[torch.Tensor, ...],
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool = False,
) -> Tuple[Optional[torch.Tensor], ...]:
    grad_output = torch.ones_like(output)

    if torch.jit.is_tracing():  # type: ignore
        err_msg = "ppe.onnx.grad() can only be used in conjunction " + \
            "with export functions under ppe.onnx"
        assert hasattr(_grad_state, "n_grad_call"), err_msg
        n_grad_call = _grad_state.n_grad_call
        _grad_state.n_grad_call += 1

        output_name = f"Gradient_y_{n_grad_call}"
        output = as_output(output_name, output)

        assert only_inputs, "only_inputs=False case is not supported now"

        input_names = []
        inputs_l = list(inputs)
        for i, input in enumerate(inputs):
            input_name = f"Gradient_x_{i}_{n_grad_call}"
            input_names.append(input_name)
            inputs_l[i] = as_output(input_name, input, add_identity=False)

        class _Gradient(torch.autograd.Function):
            @staticmethod
            def forward(  # type: ignore
                ctx: Any,
                output: torch.Tensor,
                grad_output: Optional[torch.Tensor],
                *inputs: Tuple[torch.Tensor, ...],
            ) -> Tuple[Optional[torch.Tensor], ...]:
                @torch.jit.script
                def _grad(  # type: ignore
                    output: torch.Tensor,
                    inputs: List[torch.Tensor],
                    grad_output: Optional[torch.Tensor],
                    retain_graph: Optional[bool],
                    create_graph: bool,
                    allow_unused: bool,
                ):
                    return torch.autograd.grad(
                        outputs=[output],
                        inputs=inputs,
                        grad_outputs=[grad_output],  # type: ignore
                        retain_graph=retain_graph,
                        create_graph=create_graph,
                        allow_unused=allow_unused,
                    )
                return tuple(_grad(
                    output,
                    list(inputs),
                    grad_output,
                    retain_graph,
                    create_graph,
                    allow_unused,
                ))

            @staticmethod
            def symbolic(g, output, grad_output, *inputs):  # type: ignore
                return g.op(
                    "ai.onnx.preview::Gradient",
                    *inputs,
                    xs_s=input_names,
                    zs_s=[],
                    y_s=output_name,
                    outputs=len(input_names),
                )
        return _Gradient.apply(output, grad_output, *inputs_l)  # type: ignore
    else:
        return torch.autograd.grad(
            outputs=output,
            inputs=inputs,
            grad_outputs=grad_output,
            retain_graph=retain_graph,
            create_graph=create_graph,
            only_inputs=only_inputs,
            allow_unused=allow_unused,
        )
