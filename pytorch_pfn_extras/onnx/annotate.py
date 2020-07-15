import functools

import torch
import torch.onnx.symbolic_registry as sym_reg
import torch.onnx.utils


class _AnnotationInit(object):

    def __init__(self):

        self.wrap_func_name = 'tou_wrapped_forward_'
        self.len_wrap_func_name = len(self.wrap_func_name)
        self.opname_suffix = '_tou'

        self.current_tracked_id = ''
        self.attrs_map = {}  # k=tracked_id, v=annotated attrs
        self.counter = 0  # global counter for each annotation

    def setup(self, opset_ver):
        # dryrun to register every aten ops
        sym_reg.register_version('', opset_ver)
        self.opset_ver = opset_ver

    def __enter__(self):
        self.original_kind = torch._C.Node.kind
        self.original_newnode = torch.onnx.utils._newNode

        def do_kind(_self):
            kind = self.original_kind(_self)
            if not kind.startswith('aten::'):
                return kind

            # If the target has passed wrapped forward function,
            # means the target has called with annotation.
            tracked_id = self._get_tracked_id(_self.sourceRange())
            if tracked_id == '':
                return kind

            # If the target has called with annotation,
            # this kind method returns with original suffix.
            # then ONNX converter is also required to register the new name.
            _, opname = kind.split('::')
            original_op = sym_reg._registry[('', self.opset_ver)][opname]
            wrapped_op = self._get_op(original_op, tracked_id)
            sym_reg._registry[
                ('', self.opset_ver)][opname+self.opname_suffix] = wrapped_op

            return kind + self.opname_suffix

        torch._C.Node.kind = do_kind

        torch.onnx.utils._newNode = self._get_new_node(self.original_newnode)

    def __exit__(self, type, value, traceback):
        torch._C.Node.kind = self.original_kind
        torch.onnx.utils._newNode = self.original_newnode

    def _get_tracked_id(self, source_range):
        # if not found, return empty string
        find_idx = source_range.find(self.wrap_func_name)
        if find_idx <= 0:
            return ''
        start_idx = find_idx + self.len_wrap_func_name
        next_ub_idx = source_range[start_idx:].find('_')
        tracked_id = source_range[start_idx:start_idx+next_ub_idx]
        assert tracked_id.isdigit()
        return tracked_id

    def _get_op(self, op, tracked_id):

        class _Op(object):
            # Original operator has strict ONNX foramt checker
            # and cannot add invalid kwargs because of the checker.
            # Actually this _Op is a state for tracked or not within
            # making new node.

            def __init__(_self, op, tracked_id):
                _self.op = op
                _self.tracked_id = tracked_id

            def __call__(_self, *args, **kwargs):
                self.current_tracked_id = _self.tracked_id
                return _self.op(*args, **kwargs)

        return _Op(op, tracked_id)

    def _get_new_node(self, new_node):

        class _NewNode(object):
            # if self.current_track_id is not empty, means the target
            # operator is annotated. original new_node add kwargs to
            # ONNX attributes.

            # PyTorch's ToONNX flow is supposed that:
            # n.kind() --> choose converter --> op --> new_node

            def __init__(_self, new_node):
                _self.new_node = new_node

            def __call__(_self, *args, **kwargs):
                if self.current_tracked_id != '':
                    kwargs.update(self.attrs_map[self.current_tracked_id])
                    self.current_tracked_id = ''
                return _self.new_node(*args, **kwargs)

        return _NewNode(new_node)


_annotation_init = _AnnotationInit()


def init_annotate(opset_ver):
    _annotation_init.setup(opset_ver)
    return _annotation_init


class _Annotation(object):

    def __init__(self, target, **attrs):
        self.target = target
        self.attrs = self._make_valid_attrs(**attrs)
        self.hook_count = _annotation_init.counter
        _annotation_init.counter += 1

    def _make_valid_attrs(self, **attrs):
        valid_attrs = {}

        # TODO(tanakad): use torch utils, maybe torch has more accurate one
        def get_key(k, v):
            if isinstance(v, int):
                return k+'_i'
            if isinstance(v, float):
                return k+'_f'
            if isinstance(v, str):
                return k+'_s'
            raise RuntimeError(
                'tou annotation does not support {}'.format(type(v)))

        for k, v in attrs.items():
            if isinstance(v, (tuple, list)):
                assert len(v) >= 1
                new_k = get_key(k, v[0])
            else:
                new_k = get_key(k, v)
            valid_attrs[new_k] = v
        return valid_attrs

    def __enter__(self):
        _annotation_init.attrs_map[str(self.hook_count)] = self.attrs

        self.original_forward = self.target.forward

        # Make wrapped forward method with dynamic name
        # to call other methods, use functools to make partial application
        # By calling this wrapped forward, PyTorch's tracer tracked
        # this function in source history and enable to judge annotated or not.
        created_func = {}
        fn_name = '{}{}_'.format(
            _annotation_init.wrap_func_name, self.hook_count)
        wrapped_forward_code = """def {}(fn, *args, **kwargs):
            ret = fn(*args, **kwargs)
            return ret""".format(fn_name)
        exec(wrapped_forward_code, {}, created_func)
        wrapped_forward = functools.partial(
            created_func[fn_name], self.original_forward)

        self.target.forward = wrapped_forward

    def __exit__(self, type, value, traceback):
        self.target.forward = self.original_forward


def annotate(target, **attrs):
    """Annotation parameters to the target function.

    Usage:

    >>> class Net(nn.Module):
    ...     def __init__(self):
    ...         super(Net, self).__init__()
    ...         self.conv = nn.Conv2d(1, 6, 3)
    ...         self.conv2 = nn.Conv2d(6, 12, 3)
    ...     def forward(self, x):
    ...         with tou.annotate(self.conv, key='value'):
    ...             h = self.conv(x)
    ...         h = self.conv2(h)
    ...         return h

    Use this annotate function under with statement, then the first Conv
    operator will be emit with original attributes. Attention that must set
    the target function to trace from TorchIR.

    Output ONNX is invalid because attributes are customized, so
    set ``enable_onnx_checker=False`` on exporting.

    Args:
        target (nn.Module): target function with annotation
        attrs (dict): annotation parameters
    """
    return _Annotation(target, **attrs)
