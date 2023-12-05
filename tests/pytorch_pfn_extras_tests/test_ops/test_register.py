from typing import List

import torch

from functorch.compile import make_boxed_func
from torch._dynamo.backends.common import aot_autograd


from pytorch_pfn_extras import ops


def _get_function_nodes(fx_module: torch.fx.GraphModule) -> List[torch.fx.Node]:
    return [node for node in fx_module.graph.nodes if node.op == "call_function"]


def test_register():
    def test(a):
        return a * 2

    def test_bwd(g, a):
        return g

    def test_meta(a):
        return torch.empty_like(a)

    def test_bwd_meta(g, a):
        return torch.empty_like(a)

    fwd_op = ops.OpDesc(test, test_meta, "(Tensor a) -> Tensor")
    bwd_op = ops.OpDesc(test_bwd, test_bwd_meta, "(Tensor g, Tensor a) -> Tensor")
    ops.register("test", fwd_op, bwd_op)

    class TestModule(torch.nn.Module):
        def forward(self, a):
            # Call the custom function
            return torch.ops.ppe.test(a)

    found_fwd_op = False
    found_bwd_op = False

    # Detect the custom ops
    def fwd_compiler_fn(fx_module: torch.fx.GraphModule, _):
        nonlocal found_fwd_op
        function_nodes = _get_function_nodes(fx_module)
        assert len(function_nodes) == 1
        found_fwd_op = function_nodes[0].target is torch.ops.ppe.test_fwd.default
        return make_boxed_func(fx_module)

    def bwd_compiler_fn(fx_module: torch.fx.GraphModule, _):
        nonlocal found_bwd_op
        function_nodes = _get_function_nodes(fx_module)
        assert len(function_nodes) == 1
        found_bwd_op = function_nodes[0].target is torch.ops.ppe.test_bwd.default
        return make_boxed_func(fx_module)

    aot_backend = aot_autograd(  # type: ignore[no-untyped-call]
        fw_compiler=fwd_compiler_fn,
        bw_compiler=bwd_compiler_fn,
    )
    m = TestModule()
    module_opt = torch.compile(m, fullgraph=True, backend=aot_backend)
    shape = [1, 16, 2048, 128]
    x = torch.ones(shape, requires_grad=True)
    y = module_opt(x)
    y.sum().backward()
    assert found_fwd_op
    assert found_bwd_op
