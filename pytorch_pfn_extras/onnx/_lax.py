"""
This file provides APIs to define control-flow operators (e.g., onnx::Loop and onnx::If)
when using `ppe.onnx.export`.
`torch.jit` records only `first loop` during tracing, and ppe.onnx inserts control-flow
operators after exporting ONNX.

APIs are almost same as [jax.lax](https://jax.readthedocs.io/en/latest/jax.lax.html).
"""

import torch
import threading
import onnx
import onnx.helper
from typing import Generator, Callable, Any, List, Tuple, Union, Dict
from contextlib import contextmanager

from pytorch_pfn_extras.onnx._as_output import as_output


_lax_state = threading.local()

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
    # `n_call` attribute is to avoid duplicate name created by as_output
    _lax_state.n_call = 0
    _lax_state.input_for_postproc = {}
    _lax_state.ignore_trace = False
    try:
        yield
    finally:
        _lax_state.ignore_trace = None
        _lax_state.n_call = None
        _lax_state.input_for_postproc = None


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


State = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


def _as_tuple(val: State) -> Tuple[torch.Tensor, ...]:
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
    if not torch.jit.is_tracing():  # type: ignore[no-untyped-call]
        return False
    err_msg = "functions in ppe.onnx.lax can only be used in conjunction " + \
        "with export functions under ppe.onnx"
    assert hasattr(_lax_state, "ignore_trace"), err_msg
    if hasattr(_lax_state, "ignore_trace") and _lax_state.ignore_trace:
        return False
    return True


def _create_and_register_postproc(
        create_postproc: Callable[[int], Dict[str, Any]]
) -> Dict[str, Any]:
    n_call = _lax_state.n_call
    for_postproc = create_postproc(n_call)
    _lax_state.n_call += 1
    _lax_state.input_for_postproc[n_call] = for_postproc
    return for_postproc


@contextmanager
def ignore_trace() -> Generator[None, None, None]:
    try:
        prev = _lax_state.ignore_trace
        _lax_state.ignore_trace = True
        yield
    finally:
        _lax_state.ignore_trace = prev


def fori_loop(
    lower: int,
    upper: int,
    body_fn: Callable[[torch.Tensor, State], State],
    init_val: State,
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

        n_val = len(_as_tuple(init_val))
        for_postproc = _create_and_register_postproc(lambda n_call: {
            "type": "fori_loop",
            "n_call": n_call,
            "lower": lower,
            "upper": upper,
            "it_name": f"fori_loop_it_{n_call}",
            "init_val_names": [
                f"fori_loop_prev_state_{n_call}_{i}" for i in range(n_val)
            ],
            "val_names": [f"fori_loop_state_{n_call}_{i}" for i in range(n_val)],
            "val_dtypes": [
                _torch_dtype_to_onnx_dtype_dict[v.dtype] for v in _as_tuple(init_val)
            ],
        })

        # use dummy output to return the correct outputs
        with ignore_trace():
            actual = _run(lower, upper, init_val)

        # trace first iteration
        it = torch.full(size=(), fill_value=lower, dtype=torch.int64)
        it = as_output(for_postproc["it_name"], it)
        init_val = _apply(
            init_val, lambda i, val: as_output(for_postproc["init_val_names"][i], val)
        )
        val = body_fn(it, init_val)
        val = _apply(val, lambda i, val: as_output(for_postproc["val_names"][i], val))
        out: List[torch.Tensor] = [
            _DummyOpForControlFlow.apply(v, act)  # type: ignore[no-untyped-call]
            for v, act in zip(_as_tuple(val), _as_tuple(actual))
        ]
        if is_tensor_state:
            return out[0]
        else:
            return tuple(out)
    else:
        return _run(lower, upper, init_val)


def while_loop(
        cond_fn: Callable[[State], torch.Tensor],
        body_fn: Callable[[State], State],
        init_val: State,
) -> State:
    def _run() -> State:
        val = init_val
        while cond_fn(val):
            val = body_fn(val)
        return val

    if _trace():  # type: ignore
        is_tensor_state = isinstance(init_val, torch.Tensor)

        n_val = len(_as_tuple(init_val))
        while_postproc = _create_and_register_postproc(lambda n_call: {
            "type": "while_loop",
            "n_call": n_call,
            "cond_name": f"while_loop_cond_{n_call}",
            "cond_out_name": f"while_loop_cond_out_{n_call}",
            "init_val_names": [
                f"while_loop_prev_state_{n_call}_{i}" for i in range(n_val)
            ],
            "val_names": [f"while_loop_state_{n_call}_{i}" for i in range(n_val)],
            "val_dtypes": [
                _torch_dtype_to_onnx_dtype_dict[v.dtype] for v in _as_tuple(init_val)
            ],
        })

        # use dummy output to return the correct outputs
        with ignore_trace():
            actual = _run()

        # trace first iteration
        """
        while cond_fn(val):
           val = body_fn(val)
        =>
        cond_init = cond_fn(val)
        val = Loop(
            None, cond_init, lambda _, _, state: cond_fn(state), body_fn(state), val
        )
        """
        cond_init = cond_fn(init_val)
        cond_init = as_output(while_postproc["cond_name"], cond_init)
        init_val = _apply(
            init_val, lambda i, val: as_output(while_postproc["init_val_names"][i], val)
        )
        val = body_fn(init_val)
        cond = cond_fn(val)
        val = _apply(val, lambda i, val: as_output(while_postproc["val_names"][i], val))
        cond = as_output(while_postproc["cond_out_name"], cond)
        out: List[torch.Tensor] = [
            _DummyOpForControlFlow.apply(v, act)  # type: ignore[no-untyped-call]
            for v, act in zip(_as_tuple(val), _as_tuple(actual))
        ]
        if is_tensor_state:
            return out[0]
        else:
            return tuple(out)
    else:
        return _run()


def cond(
        pred: torch.Tensor,
        true_fn: Callable[[State], State],
        false_fn: Callable[[State], State],
        operands: State,
) -> State:
    def _run() -> State:
        if pred:
            return true_fn(operands)
        else:
            return false_fn(operands)

    if _trace():  # type: ignore
        is_tensor_state = isinstance(operands, torch.Tensor)

        # use dummy output to return the correct outputs
        with ignore_trace():
            actual = _run()

        n_val = len(_as_tuple(operands))
        n_out = len(_as_tuple(actual))
        cond_postproc = _create_and_register_postproc(lambda n_call: {
            "type": "cond",
            "n_call": n_call,
            "pred_name": f"cond_pred_{n_call}",
            "operand_names": [f"cond_prev_state_{n_call}_{i}" for i in range(n_val)],
            "true_names": [f"cond_prev_true_{n_call}_{i}" for i in range(n_out)],
            "false_names": [f"cond_prev_false_{n_call}_{i}" for i in range(n_out)],
            "out_names": [f"cond_prev_out_{n_call}_{i}" for i in range(n_out)],
            "out_dtypes": [
                _torch_dtype_to_onnx_dtype_dict[v.dtype] for v in _as_tuple(actual)
            ],
        })

        # trace both branches
        pred = as_output(cond_postproc["pred_name"], pred)
        operands = _apply(
            operands, lambda i, val: as_output(cond_postproc["operand_names"][i], val)
        )
        out_true = true_fn(operands)
        out_true = _apply(
            out_true, lambda i, val: as_output(cond_postproc["true_names"][i], val)
        )
        out_false = false_fn(operands)
        out_false = _apply(
            out_false, lambda i, val: as_output(cond_postproc["false_names"][i], val)
        )
        if pred:
            _out = out_true
        else:
            _out = out_false
        _out = _apply(_out, lambda i, val: as_output(cond_postproc["out_names"][i], val))
        out: List[torch.Tensor] = [
            _DummyOpForControlFlow.apply(v, act)  # type: ignore[no-untyped-call]
            for v, act in zip(_as_tuple(_out), _as_tuple(actual))
        ]
        if is_tensor_state:
            return out[0]
        else:
            return tuple(out)
    else:
        return _run()


def _make_constant_scalar(
        name: str, dtype: onnx.TensorProto.DataType, value: Any
) -> onnx.NodeProto:
    if dtype == onnx.TensorProto.STRING:
        return onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[name],
            value=onnx.helper.make_tensor(
                name="",
                data_type=dtype,
                dims=[],
                vals=[value.encode()],
            ),
        )

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


def _to_value_infos(
        names: List[str], dtypes: List[Any], shapes: List[Any]
) -> List[onnx.TensorProto]:
    return [
        onnx.helper.make_tensor_value_info(name, dtype, shape)
        for name, dtype, shape in zip(names, dtypes, shapes)
    ]


def _find_nodes(
        graph: onnx.GraphProto, in_names: List[str], out_names: List[str]
) -> Tuple[int, List[onnx.NodeProto]]:
    class HashableNode:
        def __init__(self, node: onnx.NodeProto) -> None:
            self.node = node

        def __hash__(self) -> int:
            return hash(str(self.node))

        def __eq__(self, other: Any) -> bool:
            if isinstance(other, HashableNode):
                return bool(self.node == other.node)
            return False

    nodes = set()
    cached_results: Dict[HashableNode, bool] = {}

    name_to_nodes: Dict[str, List[onnx.NodeProto]] = {}
    node_to_index = {}
    for i, node in enumerate(graph.node):
        node_to_index[HashableNode(node)] = i
        for input in node.input:
            if input not in name_to_nodes:
                name_to_nodes[input] = []
            name_to_nodes[input].append(node)

    def _find_output_node(node: HashableNode) -> bool:
        if node in cached_results:
            return cached_results[node]

        if len(set(node.node.output) & set(out_names)) != 0:
            nodes.add(node)
            cached_results[node] = True
            return True
        found = False
        for output in node.node.output:
            if output not in name_to_nodes:
                continue
            for next in name_to_nodes[output]:
                if _find_output_node(HashableNode(next)):
                    found = True
        cached_results[node] = found
        if found:
            nodes.add(node)
        return found

    for node in graph.node:
        if len(set(node.input) & set(in_names)) != 0:
            nodes.add(HashableNode(node))
            _find_output_node(HashableNode(node))
        if len(set(node.output) & set(out_names)):
            nodes.add(HashableNode(node))

    idx = max([node_to_index[node] for node in nodes])

    # sort by original order
    node_sorted = [x.node for x in nodes]
    node_sorted.sort(key=lambda node: node_to_index[HashableNode(node)])

    return idx, node_sorted


def postprocess(onnx_graph: onnx.ModelProto) -> None:
    assert hasattr(_lax_state, "input_for_postproc")
    postprocs = list(_lax_state.input_for_postproc.items())
    # Do postprocessing in reverse order to handle nested control flows
    postprocs.sort(key=lambda v: -v[0])  # type: ignore
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
            M_const_node = _make_constant_scalar(
                M_name, onnx.TensorProto.INT64, upper - lower
            )
            cond_const_node = _make_constant_scalar(
                cond_name, onnx.TensorProto.BOOL, True
            )

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
            lower_node = _make_constant_scalar(
                lower_name, onnx.TensorProto.INT64, lower
            )
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
                nodes=[lower_node, it_node] + nodes + [cond_out_node],
                name=f"ppe_lax_Loop_{n_call}_body",
                inputs=_to_value_infos(
                    [cnt_name, cond_name] + init_val_names,
                    [onnx.TensorProto.INT64, onnx.TensorProto.BOOL] + val_dtypes,
                    [(), ()] + [None] * len(init_val_names),  # type: ignore
                ),
                outputs=_to_value_infos(
                    [cond_out_name] + val_names,
                    [onnx.TensorProto.BOOL] + val_dtypes,
                    [()] + [None] * len(val_names),  # type: ignore
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
            for node in nodes:
                onnx_graph.graph.node.remove(node)
            for node in onnx_graph.graph.node:
                if it_name in node.output or it_name in node.input:
                    onnx_graph.graph.node.remove(node)
            for output in list(onnx_graph.graph.output):
                if output.name in set(val_names + init_val_names + [it_name]):
                    onnx_graph.graph.output.remove(output)
        elif for_postproc["type"] == "while_loop":
            n_call = for_postproc["n_call"]
            cond_name = for_postproc["cond_name"]
            cond_out_name = for_postproc["cond_out_name"]
            init_val_names = for_postproc["init_val_names"]
            val_names = for_postproc["val_names"]
            val_dtypes = for_postproc["val_dtypes"]

            # Create input of Loop op
            M_name = f"ppe_lax_Loop_{n_call}_M"
            # TODO use empty string to represent infinite loop
            M_const_node = _make_constant_scalar(
                M_name, onnx.TensorProto.INT64, torch.iinfo(torch.long).max
            )

            # Create loop_body
            cnt_name = f"ppe_lax_Loop_{n_call}_cnt"
            idx, nodes = _find_nodes(
                onnx_graph.graph,
                init_val_names,
                val_names + [cond_out_name],
            )
            loop_body = onnx.helper.make_graph(
                nodes=nodes,
                name=f"ppe_lax_Loop_{n_call}_body",
                inputs=_to_value_infos(
                    [cnt_name, cond_name] + init_val_names,
                    [onnx.TensorProto.INT64, onnx.TensorProto.BOOL] + val_dtypes,
                    [(), ()] + [None] * len(init_val_names),  # type: ignore
                ),
                outputs=_to_value_infos(
                    [cond_out_name] + val_names,
                    [onnx.TensorProto.BOOL] + val_dtypes,
                    [()] + [None] * len(val_names),  # type: ignore
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
            onnx_graph.graph.node.insert(idx + 2, loop_node)
            for node in nodes:
                onnx_graph.graph.node.remove(node)
            unused_outputs = set(
                val_names + init_val_names + [cond_name, cond_out_name]
            )
            for output in list(onnx_graph.graph.output):
                if output.name in unused_outputs:
                    onnx_graph.graph.output.remove(output)
        elif for_postproc["type"] == "cond":
            n_call = for_postproc["n_call"]
            pred_name = for_postproc["pred_name"]
            operand_names = for_postproc["operand_names"]
            true_names = for_postproc["true_names"]
            false_names = for_postproc["false_names"]
            out_names = for_postproc["out_names"]
            out_dtypes = for_postproc["out_dtypes"]

            # Create input of Loop op
            M_name = f"ppe_lax_Loop_{n_call}_M"
            # TODO use empty string to represent infinite loop
            M_const_node = _make_constant_scalar(
                M_name, onnx.TensorProto.INT64, torch.iinfo(torch.long).max
            )

            # Create then_branch
            then_idx, then_nodes = _find_nodes(
                onnx_graph.graph,
                operand_names,
                true_names,
            )
            then_branch = onnx.helper.make_graph(
                nodes=then_nodes,
                name=f"pee_lax_If_{n_call}_then",
                inputs=[],
                outputs=_to_value_infos(
                    true_names,
                    out_dtypes,
                    [None] * len(out_dtypes)  # type: ignore
                )
            )
            # Create else_branch
            else_idx, else_nodes = _find_nodes(
                onnx_graph.graph,
                operand_names,
                false_names,
            )
            else_branch = onnx.helper.make_graph(
                nodes=else_nodes,
                name=f"pee_lax_If_{n_call}_else",
                inputs=[],
                outputs=_to_value_infos(
                    false_names,
                    out_dtypes,
                    [None] * len(out_dtypes)  # type: ignore
                )
            )

            if_node = onnx.helper.make_node(
                name=f"ppe_lax_If_{n_call}",
                op_type="If",
                inputs=[pred_name],
                outputs=out_names,
                then_branch=then_branch,
                else_branch=else_branch,
            )
            idx = max(then_idx, else_idx)
            onnx_graph.graph.node.insert(idx, if_node)
            nodes = then_nodes + else_nodes
            for node in nodes:
                if node in onnx_graph.graph.node:
                    onnx_graph.graph.node.remove(node)
            # remove Identity from {true/false} to out
            for node in list(onnx_graph.graph.node):
                if node.op_type == "If":
                    continue
                if set(node.input) & set(true_names + false_names):
                    assert node.op_type == "Identity"
                    onnx_graph.graph.node.remove(node)
                if set(node.output) & set(out_names):
                    assert node.op_type == "Identity"
                    onnx_graph.graph.node.remove(node)
            unused_outputs = set(true_names + false_names + out_names + operand_names)
            for output in list(onnx_graph.graph.output):
                if output.name in unused_outputs:
                    onnx_graph.graph.output.remove(output)
        else:
            raise RuntimeError("Invalid lax type: " + for_postproc["type"])
