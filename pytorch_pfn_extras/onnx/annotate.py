import contextlib
import functools
import types
from typing import Any, Callable, ContextManager, Dict, Generator, List, Type, Optional

import onnx.helper
import torch
import torch.nn as nn
import torch.onnx
import pytorch_pfn_extras


class _AnnotationInit(object):

    def __init__(self) -> None:
        self.wrap_func_name = 'tou_wrapped_forward_'
        self.len_wrap_func_name = len(self.wrap_func_name)
        self.opname_suffix = '_tou'

        # k=tracked_id, v=annotated attrs
        self.attrs_map: Dict[str, Dict[str, Any]] = {}

        self.counter = 0  # global counter for each annotation

        self.anchor_func_name = 'tou_anchor_'
        self.len_anchor_func_name = len(self.anchor_func_name)

        # k=global count, v=number of node
        self.anchored_node_count: Dict[str, str] = {}

    def setup(self, model: nn.Module, opset_ver: int) -> None:
        self._model: Optional[nn.Module] = model
        # dryrun to register every aten ops
        if not pytorch_pfn_extras.requires("1.13.0"):
            import pytorch_pfn_extras.onnx.symbolic_registry as sym_reg

            sym_reg.register_version('', opset_ver)  # type: ignore[no-untyped-call,attr-defined]
        self.opset_ver = opset_ver

    @property
    def model(self) -> nn.Module:
        assert self._model is not None
        return self._model

    def __enter__(self) -> '_AnnotationInit':
        return self

    def __exit__(
            self,
            type: Optional[Type[BaseException]],
            value: Optional[BaseException],
            traceback: Optional[types.TracebackType],
    ) -> None:
        self._model = None
        self.attrs_map = {}
        self.counter = 0
        self.anchored_node_count = {}

    def set_annotate(self, onnx_graph: Any) -> Any:
        for node in onnx_graph.graph.node:
            find_idx = node.doc_string.find(self.wrap_func_name)
            if find_idx < 0:
                continue
            self._edit_attr(node, find_idx)

        return onnx_graph

    def _edit_attr(self, node: Any, found_idx: int) -> None:
        start_idx = found_idx + self.len_wrap_func_name
        next_ub_idx = node.doc_string[start_idx:].find('_')
        tracked_id = node.doc_string[start_idx:start_idx + next_ub_idx]
        assert tracked_id.isdigit()

        attrs = [onnx.helper.make_attribute(k, v) for k, v in
                 self.attrs_map[tracked_id].items()]
        node.attribute.extend(attrs)

    def reorg_anchor(self, onnx_graph: Any) -> Any:
        # inspect all node before editing anchors, need to edit previous node
        nodes_by_input: Dict[int, List[Any]] = {}  # for re-connect nodes
        anchor_nodes = {}  # extract anchors
        dummy_constant_ids = []  # remove them after
        for node in onnx_graph.graph.node:
            for input_id in node.input:
                nodes_by_input.setdefault(input_id, []).append(node)

            find_idx = node.doc_string.find(self.anchor_func_name)
            if find_idx >= 0:
                anchor_nodes[node.name] = find_idx
                dummy_constant_ids.append(node.input[1])

        # cleanup anchor node
        reorged_nodes = []
        for node in onnx_graph.graph.node:
            if node.op_type == 'Constant':
                if node.output[0] not in dummy_constant_ids:
                    reorged_nodes.append(node)
                continue

            if node.name in anchor_nodes:
                can_remove = self._edit_anchor(node, anchor_nodes[node.name])
                if can_remove:
                    # re-connect node
                    out_id = node.output[0]
                    for connected_node in nodes_by_input[out_id]:
                        for i, node_input in enumerate(connected_node.input):
                            if node_input == out_id:
                                connected_node.input[i] = node.input[0]
                    continue

                reorged_nodes.append(node)
            else:
                reorged_nodes.append(node)

        del onnx_graph.graph.node[:]
        onnx_graph.graph.node.extend(reorged_nodes)

        return onnx_graph

    def _edit_anchor(self, node: Any, found_idx: int) -> bool:
        # return enable to delete or not
        assert node.op_type == 'Add'

        start_idx = found_idx + self.len_anchor_func_name
        end_idx = node.doc_string[start_idx:].find('__')
        track_info = node.doc_string[start_idx:start_idx + end_idx].split('_')
        assert len(track_info) == 3
        start_end, tracked_id, node_num = track_info

        node_total = self.anchored_node_count[tracked_id]
        if start_end == 'e' and node_total != node_num:
            return True

        attrs = [onnx.helper.make_attribute(k, v) for k, v in
                 self.attrs_map[tracked_id].items()]
        node.attribute.extend(attrs)
        node.op_type = 'Identity'
        node.name = 'Anchor_{}_{}'.format(
            tracked_id, 'start' if start_end == 's' else 'end')
        del node.input[1]
        return False


_annotation_init = _AnnotationInit()


def init_annotate(model: nn.Module, opset_ver: int) -> _AnnotationInit:
    _annotation_init.setup(model, opset_ver)
    return _annotation_init


class _Annotation(object):

    def __init__(self, **attrs: Any) -> None:
        self.attrs = attrs
        self.hook_count = _annotation_init.counter
        _annotation_init.counter += 1
        self.original_forwards: Dict[str, Callable[..., Any]] = {}

    def __enter__(self) -> None:
        _annotation_init.attrs_map[str(self.hook_count)] = self.attrs

        # Make wrapped forward method with dynamic name
        # to call other methods, use functools to make partial application
        # By calling this wrapped forward, PyTorch's tracer tracked
        # this function in source history and enable to judge annotated or not.
        created_func: Dict[str, Any] = {}
        fn_name = '{}{}_'.format(
            _annotation_init.wrap_func_name, self.hook_count)
        wrapped_forward_code = """def {}(fn, *args, **kwargs):
            ret = fn(*args, **kwargs)
            return ret""".format(fn_name)
        exec(wrapped_forward_code, {}, created_func)

        for name, child_module in _annotation_init.model.named_children():
            original_forward = child_module.forward
            self.original_forwards[name] = original_forward
            wrapped_forward = functools.partial(
                created_func[fn_name], original_forward)
            child_module.forward = wrapped_forward  # type: ignore[assignment]

    def __exit__(
            self,
            type: Optional[Type[BaseException]],
            value: Optional[BaseException],
            traceback: Optional[types.TracebackType],
    ) -> None:
        for name, child_module in _annotation_init.model.named_children():
            child_module.forward = self.original_forwards[name]  # type: ignore


@contextlib.contextmanager
def _nullcontext() -> Generator[None, None, None]:
    # contextlib.nullcontext equivalent, needed for Python 3.6 support.
    yield


def annotate(**attrs: Any) -> ContextManager[None]:
    """Annotation parameters to the target function.

    Usage:

    >>> class Net(nn.Module):
    ...     def __init__(self):
    ...         super(Net, self).__init__()
    ...         self.conv = nn.Conv2d(1, 6, 3)
    ...         self.conv2 = nn.Conv2d(6, 12, 3)
    ...     def forward(self, x):
    ...         with pytorch_pfn_extras.onnx.annotate(key='value'):
    ...             h = self.conv(x)
    ...         h = self.conv2(h)
    ...         return h

    Use this annotate function under with statement, then the first Conv
    operator will be emit with customized attributes. Customized attributes
    are invalid for ONNX format, so pay attention that some ONNX runtimes
    cannot run the output ONNX graph.

    This annotation is enabled with either
    ``pytorch_pfn_extras.onnx.export_testcase`` or
    ``pytorch_pfn_extras.onnx.export``.

    Args:
        attrs (dict): annotation parameters
    """
    if torch.onnx.is_in_onnx_export():  # type: ignore[no-untyped-call]
        return _Annotation(**attrs)
    return _nullcontext()


def apply_annotation(fn: Callable[..., Any], *args: Any, **attrs: Any) -> Any:
    """Annotation applier to the target function

    Usage:

    >>> class Net(nn.Module):
    ...     def __init__(self):
    ...         super(Net, self).__init__()
    ...         self.conv = nn.Conv2d(1, 6, 3)
    ...         self.conv2 = nn.Conv2d(6, 12, 3)
    ...     def forward(self, x):
    ...         def _conv(x):
    ...             h = self.conv(x)
    ...             return torch.relu(h)
    ...         h = pytorch_pfn_extras.onnx.apply_annotation(
    ...             _conv, key='value')
    ...         h = self.conv2(h)
    ...         return h

    Annotate into all operators emitted from the target function even if
    included not ``nn.Module`` function. On the above code, the first Conv and
    ReLu operator will be emit with customized attributes. Customized
    attributes are invlid for ONNX format, so pay attention that some ONNX
    runtimes cannot run the output ONNX graph.

    This applier is enabled with either
    ``pytorch_pfn_extras.onnx.export_testcase`` or
    ``pytorch_pfn_extras.onnx.export``.

    Args:
        fn (func): the target function to be annotated, ``args`` is used for
            this function. Cannot pass ``kwargs`` for the function.
        args (tuple): arguments for the target function
        attrs (dict): annotation paramters
    """

    class _DoFunction(nn.Module):

        def __init__(self, fn: Callable[..., Any], *args: Any) -> None:
            super(_DoFunction, self).__init__()  # type: ignore[no-untyped-call]
            self.fn = fn
            self.args = args

        def forward(self) -> Any:
            return self.fn(*args)

    wrapped_fn = _DoFunction(fn)
    wrapped_fn_id = str(id(wrapped_fn))
    setattr(_annotation_init.model, wrapped_fn_id, wrapped_fn)
    try:
        with annotate(**attrs):
            ret = wrapped_fn()
    finally:
        delattr(_annotation_init.model, wrapped_fn_id)
    return ret


class _Anchor(_Annotation):

    def __init__(self, **attrs: Any) -> None:
        super(_Anchor, self).__init__(**attrs)
        self.called_count = -1
        self.started = False

    def _get_anchor_func(self, start_end: str = 's') -> Any:
        self.called_count += 1
        created_func: Dict[str, Any] = {}

        # wrapped name + start/end + global count + internal count
        fn_name = '{}{}_{}_{}__'.format(
            _annotation_init.anchor_func_name, start_end,
            self.hook_count, self.called_count)
        wrapped_anchor_code = """def {}(x, y):
            return x + y""".format(fn_name)
        exec(wrapped_anchor_code, {}, created_func)

        return created_func[fn_name]

    def __enter__(self) -> None:
        _annotation_init.attrs_map[str(self.hook_count)] = self.attrs

        def do_forward(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:

            def dummy_anchor(fn: Callable[..., Any], x: Any) -> Any:
                zero = torch.zeros((1,), dtype=x.dtype)
                return fn(x, zero)

            if not self.started:
                assert len(args) > 0
                arg0 = args[0]
                dfunc = self._get_anchor_func()
                if isinstance(arg0, torch.Tensor):
                    arg0 = dummy_anchor(dfunc, arg0)
                elif isinstance(arg0, (list, tuple)):
                    assert len(arg0) > 0
                    arg00 = arg0[0]
                    dfunc = self._get_anchor_func()
                    arg00 = dummy_anchor(dfunc, arg00)
                    if isinstance(arg0, list):
                        arg0 = [arg00] + arg0[1:]
                    else:
                        arg0 = (arg00,) + arg0[1:]
                else:
                    raise RuntimeError(
                        'type {} is not supported for anchor input'.format(
                            type(arg0)))
                args = (arg0,) + args[1:]
                self.started = True

            out = fn(*args, **kwargs)

            dfunc = self._get_anchor_func(start_end='e')
            if isinstance(out, torch.Tensor):
                return dummy_anchor(dfunc, out)
            elif isinstance(out, (list, tuple)):
                assert len(out) > 0
                out0 = dummy_anchor(dfunc, out[0])
                if isinstance(out, list):
                    return [out0] + out[1:]
                else:
                    return (out0,) + out[1:]
            else:
                raise RuntimeError(
                    'type {} is not supported for anchor output'.format(
                        type(out)))

        for name, child_module in _annotation_init.model.named_children():
            original_forward = child_module.forward
            self.original_forwards[name] = original_forward

            wrapped_forward = functools.partial(do_forward, original_forward)
            child_module.forward = wrapped_forward  # type: ignore[assignment]

    def __exit__(
            self,
            type: Optional[Type[BaseException]],
            value: Optional[BaseException],
            traceback: Optional[types.TracebackType],
    ) -> None:
        for name, child_module in _annotation_init.model.named_children():
            child_module.forward = self.original_forwards[name]  # type: ignore

        _annotation_init.anchored_node_count[str(self.hook_count)] = \
            str(self.called_count)


def scoped_anchor(**attrs: Any) -> ContextManager[None]:
    """Add anchor node to the scoped modules

    Usage:

    >>> class Net(nn.Module):
    ...     def __init__(self):
    ...         super(Net, self).__init__()
    ...         self.conv = nn.Conv2d(1, 6, 3)
    ...         self.conv2 = nn.Conv2d(6, 12, 3)
    ...     def forward(self, x):
    ...         with pytorch_pfn_extras.onnx.scoped_anchor(key='value'):
    ...             h = self.conv(x)
    ...         h = self.conv2(h)
    ...         return h

    Use this scoped anchoring under with statement, then dummy Identity nodes
    are added before/after the first Conv operator with customized attributes.

    This anchoring is triggered by ``nn.Module`` applying function, cannot
    use this with ``torch.*`` functions.

    This annotation is enabled with either
    ``pytorch_pfn_extras.onnx.export_testcase`` or
    ``pytorch_pfn_extras.onnx.export``.

    Args:
        attrs (dict): annotation parameters
    """
    if torch.onnx.is_in_onnx_export():  # type: ignore[no-untyped-call]
        return _Anchor(**attrs)
    return _nullcontext()
