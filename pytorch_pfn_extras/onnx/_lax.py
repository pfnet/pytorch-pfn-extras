# equivalent to [jax.lax](https://jax.readthedocs.io/en/latest/jax.lax.html)
import torch
import threading
import onnx
import onnx.helper
from typing import Generator, Callable, Any, List, Tuple, Union
from contextlib import contextmanager

from pytorch_pfn_extras.onnx._as_output import as_output


_lax_state = threading.local()

# copy from https://github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L367
_torch_dtype_to_onnx_dtype_dict = {
    torch.bool: onnx.TensorProto.BOOL,
    torch.uint8: onnx.TensorProto.UINT8,
    torch.int8: onnx.TensorProto.INT8,
    torch.int16: onnx.TensorProto.INT16,
    torch.int32: onnx.TensorProto.INT32,
    torch.int64: onnx.TensorProto.INT64,
    torch.float16: onnx.TensorProto.FLOAT16,
    torch.float32: onnx.TensorProto.FLOAT,
    torch.float64: onnx.TensorProto.DOUBLE,
    torch.complex64: onnx.TensorProto.COMPLEX64,
    torch.complex128: onnx.TensorProto.COMPLEX128,
}


@contextmanager
def init_lax_state() -> Generator[None, None, None]:
    _lax_state.n_call = 0
    _lax_state.input_for_postproc = {}
    _lax_state.ignore_trace = False
    try:
        yield
    finally:
        _lax_state.ignore_trace = None
        _lax_state.n_call = None
        _lax_state.input_for_postproc = None


# Add Identity function to prevent constant folding in torch.onnx
class _ExplicitIdentity(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        it: torch.Tensor,
    ) -> torch.Tensor:
        return it

    @staticmethod
    def symbolic(g, it):  # type: ignore
        return g.op("Identity", it)


class _DummyOpForControlFlow(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        init_val: torch.Tensor,
        actual_val: torch.Tensor,
    ) -> torch.Tensor:
        return actual_val

    @staticmethod
    def symbolic(g, init_val, actual_val):  # type: ignore
        return init_val


State = Union[torch.Tensor, Tuple[torch.Tensor]]


def _as_tuple(val: State) -> Tuple[torch.Tensor]:
    if isinstance(val, torch.Tensor):
        return tuple([val])
    else:
        assert isinstance(val, tuple)
        return val


def _apply(val: State, f: Callable[[int, torch.Tensor], torch.Tensor]) -> State:
    if isinstance(val, torch.Tensor):
        return f(0, val)
    else:
        assert isinstance(val, tuple)
        return tuple([f(i, v) for i, v in enumerate(val)])


def _trace() -> bool:
    if not torch.jit.is_tracing():
        return False
    if hasattr(_lax_state, "ignore_trace") and _lax_state.ignore_trace:
        return False
    return True


def fori_loop(
    lower: int, upper: int, body_fn: Callable[[torch.Tensor, State], State], init_val: State
) -> State:
    def _run(lower: int, upper: int, init_val: State) -> State:
        val = init_val
        for i in range(lower, upper):
            it = torch.full(size=(), fill_value=i)
            val = body_fn(it, val)
        return val

    if _trace():  # type: ignore
        if lower >= upper:
            return init_val
        is_tensor_state = isinstance(init_val, torch.Tensor)

        err_msg = "ppe.onnx.jax.fori_loop() can only be used in conjunction " + \
            "with export functions under ppe.onnx"
        assert hasattr(_lax_state, "n_call"), err_msg
        n_call = _lax_state.n_call
        n_val = len(_as_tuple(init_val))
        for_postproc = {
            "type": "fori_loop",
            "n_call": n_call,
            "lower": lower,
            "upper": upper,
            "it_name": f"fori_loop_it_{n_call}",
            "init_val_names": [f"fori_loop_prev_state_{n_call}_{i}" for i in range(n_val)],
            "val_names": [f"fori_loop_state_{n_call}_{i}" for i in range(n_val)],
            "val_dtypes": [_torch_dtype_to_onnx_dtype_dict[v.dtype] for v in _as_tuple(init_val)],
        }
        _lax_state.n_call += 1
        _lax_state.input_for_postproc[n_call] = for_postproc

        # use dummy output to return the correct outputs
        try:
            prev = _lax_state.ignore_trace
            _lax_state.ignore_trace = True
            actual = _run(lower, upper, init_val)
        finally:
            _lax_state.ignore_trace = prev

        # trace first iteration
        it = torch.full(size=(), fill_value=lower, dtype=torch.int64)
        it = _ExplicitIdentity.apply(it)
        it = as_output(for_postproc["it_name"], it)
        init_val = _apply(init_val, lambda i, val: _ExplicitIdentity.apply(val))
        init_val = _apply(init_val, lambda i, val: as_output(for_postproc["init_val_names"][i], val))
        val = body_fn(it, init_val)
        val = _apply(val, lambda i, val: as_output(for_postproc["val_names"][i], val))
        val = _apply(val, lambda i, val: _ExplicitIdentity.apply(val))
        out = [
            _DummyOpForControlFlow.apply(v, act)
            for v, act in zip(_as_tuple(val), _as_tuple(actual))
        ]
        if is_tensor_state:
            return out[0]
        else:
            return tuple(out)
    else:
        return _run(lower, upper, init_val)


def _make_constant_scalar(name: str, dtype: onnx.TensorProto.DataType, value: Any) -> onnx.NodeProto:
    return onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=onnx.helper.make_tensor(
            name="",
            data_type=dtype,
            dims=[],
            vals=[value],
        ),
    )


def _to_value_infos(names: List[str], dtypes: List[Any], shapes: List[Any]) -> List[onnx.TensorProto]:
    return [
        onnx.helper.make_tensor_value_info(name, dtype, shape)
        for name, dtype, shape in zip(names, dtypes, shapes)
    ]


def _find_nodes(graph: onnx.GraphProto, in_names: List[str], out_names: List[str]) -> Tuple[int, List[onnx.NodeProto]]:
    class HashableNode:
        def __init__(self, node: onnx.NodeProto) -> None:
            self.node = node

        def __hash__(self) -> int:
            return hash(str(self.node))

        def __eq__(self, other: Any) -> bool:
            if isinstance(other, HashableNode):
                return self.node == other.node
            return False

    nodes = set()
    visited = set()

    name_to_nodes = {}
    node_to_index = {}
    for i, node in enumerate(graph.node):
        node_to_index[HashableNode(node)] = i
        for input in node.input:
            if input not in name_to_nodes:
                name_to_nodes[input] = []
            name_to_nodes[input].append(node)

    def _find_output_node(node: HashableNode) -> bool:
        if node in visited:
            return False

        visited.add(node)
        if len(set(node.node.output) & set(out_names)) != 0:
            nodes.add(node)
            return True
        for output in node.node.output:
            found = False
            if output not in name_to_nodes:
                continue
            for next in name_to_nodes[output]:
                if _find_output_node(HashableNode(next)):
                    found = True
                    break
            if found:
                nodes.add(node)
                return True
        return False

    for node in graph.node:
        if len(set(node.input) & set(in_names)) != 0:
            nodes.add(HashableNode(node))
            _find_output_node(HashableNode(node))
        if len(set(node.output) & set(out_names)):
            nodes.add(HashableNode(node))

    idx = min([node_to_index[node] for node in nodes])

    # sort by original order
    node_sorted = [x.node for x in nodes]
    node_sorted.sort(key=lambda node: node_to_index[HashableNode(node)])

    return idx, node_sorted


def postprocess(onnx_graph: onnx.ModelProto) -> None:
    assert hasattr(_lax_state, "input_for_postproc")
    postprocs = list(_lax_state.input_for_postproc.items())
    # Do postprocessing in reverse order to handle nested control flows
    postprocs.sort(key=lambda v: -v[0])
    for _, for_postproc in postprocs:
        if for_postproc["type"] == "fori_loop":
            n_call = for_postproc["n_call"]
            lower = for_postproc["lower"]
            upper = for_postproc["upper"]
            it_name = for_postproc["it_name"]
            init_val_names = for_postproc["init_val_names"]
            val_names = for_postproc["val_names"]
            val_dtypes = for_postproc["val_dtypes"]

            # Create input of Loop op
            M_name = f"ppe_lax_Loop_{n_call}_M"
            cond_name = f"ppe_lax_Loop_{n_call}_cond"
            M_const_node = _make_constant_scalar(M_name, onnx.TensorProto.INT64, upper - lower)
            cond_const_node = _make_constant_scalar(cond_name, onnx.TensorProto.BOOL, True)

            # Create loop_body
            cond_out_name = f"ppe_lax_Loop_{n_call}_cond_out"
            cond_out_node = onnx.helper.make_node(
                "Identity",
                inputs=[cond_name],
                outputs=[cond_out_name],
                name=cond_out_name + "_Identity",
            )
            cnt_name = f"ppe_lax_Loop_{n_call}_cnt"
            lower_name = f"ppe_lax_Loop_{n_call}_lower"
            lower_node = _make_constant_scalar(lower_name, onnx.TensorProto.INT64, lower)
            it_node = onnx.helper.make_node(
                "Add",
                inputs=[cnt_name, lower_name],
                outputs=[it_name],
                name=it_name + "_Add",
            )
            idx, nodes = _find_nodes(
                onnx_graph.graph,
                init_val_names + [it_name],
                val_names,
            )
            loop_body = onnx.helper.make_graph(
                nodes=[lower_node, it_node, ] + nodes + [cond_out_node],
                name=f"ppe_lax_Loop_{n_call}_body",
                inputs=_to_value_infos(
                    [cnt_name, cond_name] + init_val_names,
                    [onnx.TensorProto.INT64, onnx.TensorProto.BOOL] + val_dtypes,
                    [(), ()] + [None] * len(init_val_names),
                ),
                outputs=_to_value_infos(
                    [cond_out_name] + val_names,
                    [onnx.TensorProto.BOOL] + val_dtypes,
                    [()] + [None] * len(val_names),
                ),
            )
            loop_node = onnx.helper.make_node(
                name=f"ppe_lax_Loop_{n_call}",
                op_type="Loop",
                inputs=[M_name, cond_name] + init_val_names,
                outputs=val_names,
                body=loop_body,
            )
            assert len(nodes) != 0
            onnx_graph.graph.node.insert(idx, M_const_node)
            onnx_graph.graph.node.insert(idx + 1, cond_const_node)
            onnx_graph.graph.node.insert(idx + 2, loop_node)
            onnx_graph.graph.node.insert(idx, _make_constant_scalar("/Constant_output_0", onnx.TensorProto.BOOL, True))
            for node in nodes:
                onnx_graph.graph.node.remove(node)
            for node in onnx_graph.graph.node:
                if it_name in node.output:
                    onnx_graph.graph.node.remove(node)
            for output in list(onnx_graph.graph.output):
                if output.name in val_names or output.name in init_val_names or output.name == it_name:
                    onnx_graph.graph.output.remove(output)
        else:
            raise RuntimeError("Invalid lax type: " + for_postproc["type"])
