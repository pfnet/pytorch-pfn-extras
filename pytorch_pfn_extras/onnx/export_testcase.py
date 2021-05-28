import datetime
import io
import itertools
import json
import os
import subprocess
import warnings

import onnx
import onnx.numpy_helper
import torch
import torch.autograd
from torch.onnx import OperatorExportTypes
from torch.onnx.symbolic_helper import _default_onnx_opset_version
from torch.onnx.utils import \
    _export as torch_export, _model_to_graph as torch_model_to_graph

from pytorch_pfn_extras.onnx.annotate import init_annotate
from pytorch_pfn_extras.onnx.strip_large_tensor import \
    LARGE_TENSOR_DATA_THRESHOLD
from pytorch_pfn_extras.onnx.strip_large_tensor import is_large_tensor
from pytorch_pfn_extras.onnx.strip_large_tensor import _strip_raw_data
from pytorch_pfn_extras.onnx.strip_large_tensor import \
    _strip_large_initializer_raw_data


def _model_to_graph_with_value_names(*args, add_value_names=True, **kwargs):
    g, p, o = torch_model_to_graph(*args, **kwargs)
    if not add_value_names:
        return g, p, o

    for n in g.nodes():
        for v in itertools.chain(n.inputs(), n.outputs()):
            if not v.debugName().isnumeric():
                continue
            old_name = v.debugName()
            new_name = 'v{}_{}'.format(old_name, n.kind().split('::')[-1])
            v.setDebugName(new_name)
            if old_name in p:
                i = p[old_name]
                del p[old_name]
                p[new_name] = i
    return g, p, o


def _export_meta(model, out_dir, strip_large_tensor_data, user_meta):
    ret = {
        'generated_at': datetime.datetime.now().isoformat(),
        'output_directory': out_dir,
        'exporter': 'torch-onnx-utils',
        'strip_large_tensor_data': strip_large_tensor_data,
    }
    if user_meta:
        ret['user_meta'] = user_meta

    try:
        git_status = subprocess.Popen(['git', 'status'],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
        git_status.communicate()

        def strip_cmd(cmd):
            with os.popen(cmd) as f:
                return f.read().strip()
        if git_status.returncode == 0:
            ret['git'] = {
                'branch': strip_cmd('git rev-parse --abbrev-ref HEAD'),
                'commit': strip_cmd('git rev-parse HEAD'),
                'remote': strip_cmd('git ls-remote --get-url origin'),
                'commit_date': strip_cmd('git show -s --format=%ci HEAD'),
            }
    except FileNotFoundError:
        pass

    return ret


def _export_util(model, args, f, **kwargs):
    """Wrap operator type to export

    Copied from torch.onnx.utils.export, to get output values.
    """
    aten = kwargs.get('aten', False)
    export_raw_ir = kwargs.get('export_raw_ir', False)
    operator_export_type = kwargs.get('operator_export_type', None)

    if aten or export_raw_ir:
        assert operator_export_type is None
        assert aten ^ export_raw_ir
        operator_export_type = OperatorExportTypes.ATEN if\
            aten else OperatorExportTypes.RAW
    elif operator_export_type is None:
        if torch.onnx.PYTORCH_ONNX_CAFFE2_BUNDLE:
            operator_export_type = OperatorExportTypes.ONNX_ATEN_FALLBACK
        else:
            operator_export_type = OperatorExportTypes.ONNX

    old_model_to_graph = torch.onnx.utils._model_to_graph
    # TODO(ecastill) _model_to_graph shouldn't be direclty overriden
    # This is a temporal workaround until a fix is introduced in PyTorch.
    try:
        torch.onnx.utils._model_to_graph = _model_to_graph_with_value_names
        return torch_export(model, args, f, _retain_param_name=True, **kwargs)
    finally:
        torch.onnx.utils._model_to_graph = old_model_to_graph


def _export(
        model, args, strip_large_tensor_data=False,
        large_tensor_threshold=LARGE_TENSOR_DATA_THRESHOLD, **kwargs):
    model.zero_grad()
    bytesio = io.BytesIO()
    opset_ver = kwargs.get('opset_version', None)
    if opset_ver is None:
        opset_ver = _default_onnx_opset_version
    strip_doc_string = kwargs.pop('strip_doc_string', True)
    with init_annotate(model, opset_ver) as ann:
        outs = _export_util(
            model, args, bytesio, strip_doc_string=False, **kwargs)
        onnx_graph = onnx.load(io.BytesIO(bytesio.getvalue()))
        onnx_graph = ann.set_annotate(onnx_graph)
        onnx_graph = ann.reorg_anchor(onnx_graph)
    if strip_doc_string:
        for node in onnx_graph.graph.node:
            node.doc_string = b''
    if strip_large_tensor_data:
        _strip_large_initializer_raw_data(onnx_graph, large_tensor_threshold)

    return onnx_graph, outs


def export(
        model, args, f, return_output=False, strip_large_tensor_data=False,
        large_tensor_threshold=LARGE_TENSOR_DATA_THRESHOLD, **kwargs):
    """Export model into ONNX Graph.

    Args:
        f: A file-like object or a string file path to be written to this
            file.
        return_output (bool): If True, return output values come from the
            model.
        strip_large_tensor_data (bool): If True, this function will strip
            data of large tensors to reduce ONNX file size for benchmarking
        large_tensor_threshold (int): If number of elements of tensor is
            larger than this value, the tensor is stripped when
            *strip_large_tensor_data* is True

    .. warning:: This function is not thread safe.

    """
    onnx_graph, outs = _export(
        model, args, strip_large_tensor_data, large_tensor_threshold,
        **kwargs)

    if hasattr(f, 'write'):
        f.write(onnx_graph.SerializeToString())
    else:
        assert isinstance(f, str)
        warnings.warn(
            'When export ONNX graph as file, "export_testcase" is '
            'strongly recommended, please consider use it instead',
            UserWarning)
        with open(f, 'wb') as fp:
            fp.write(onnx_graph.SerializeToString())

    if return_output:
        return outs


def export_testcase(
        model, args, out_dir, *, output_grad=False, metadata=True,
        model_overwrite=True, strip_large_tensor_data=False,
        large_tensor_threshold=LARGE_TENSOR_DATA_THRESHOLD,
        return_output=False, user_meta=None,
        export_torch_script=False, export_torch_trace=False, **kwargs):
    """Export model and I/O tensors of the model in protobuf format.

    Args:
        output_grad (bool or Tensor): If True, this function will output
            model's gradient with names 'gradient_%d.pb'. If set Tensor,
            use it as gradient *input*. The gradient inputs are output as
            'gradient_input_%d.pb' along with gradient.
        metadata (bool): If True, output meta information taken from git log.
        model_overwrite (bool): If False and model.onnx has already existed,
            only export input/output data as another test dataset.
        strip_large_tensor_data (bool): If True, this function will strip
            data of large tensors to reduce ONNX file size for benchmarking
        large_tensor_threshold (int): If number of elements of tensor is
            larger than this value, the tensor is stripped when
            *strip_large_tensor_data* is True
        return_output (bool): If True, return output values come from the
            model.
        export_torch_script (bool): Output model_script.pt using
            torch.jit.script
        export_torch_trace (bool): Output model_trace.pt using torch.jit.trace

    .. warning:: This function is not thread safe.

    """

    os.makedirs(out_dir, exist_ok=True)
    input_names = kwargs.pop(
        'input_names',
        ['input_{}'.format(i) for i in range(len(args))])
    assert len(input_names) == len(args)

    onnx_graph, outs = _export(
        model, args, strip_large_tensor_data, large_tensor_threshold,
        input_names=input_names, **kwargs)
    if isinstance(args, torch.Tensor):
        args = args,
    if isinstance(outs, torch.Tensor):
        outs = outs,

    # Remove unused inputs
    # - When keep_initializers_as_inputs=True, inputs contains initializers.
    #   So we have to filt initializers.
    # - model.onnx is already issued, so we can modify args here.
    initializer_names = [init.name for init in onnx_graph.graph.initializer]
    used_input_index_list = []
    for used_input in onnx_graph.graph.input:
        if used_input.name not in initializer_names:
            used_input_index_list.append(input_names.index(used_input.name))
    input_names = [input_names[i] for i in used_input_index_list]
    args = [args[i] for i in used_input_index_list]

    output_path = os.path.join(out_dir, 'model.onnx')
    is_on_memory = True
    if model_overwrite or (not os.path.isfile(output_path)):
        is_on_memory = False
        with open(output_path, 'wb') as fp:
            fp.write(onnx_graph.SerializeToString())

    def write_to_pb(f, tensor, name=None):
        array = tensor.detach().cpu().numpy()
        with open(f, 'wb') as fp:
            t = onnx.numpy_helper.from_array(array, name)
            if (strip_large_tensor_data
                    and is_large_tensor(t, large_tensor_threshold)):
                _strip_raw_data(t)
            fp.write(t.SerializeToString())

    if export_torch_script:
        pt_script_path = os.path.join(out_dir, 'model_script.pt')
        if model_overwrite or (not os.path.isfile(pt_script_path)):
            torch.jit.script(model).save(pt_script_path)

    if export_torch_trace:
        pt_trace_path = os.path.join(out_dir, 'model_trace.pt')
        if model_overwrite or (not os.path.isfile(pt_trace_path)):
            torch.jit.trace(model, args).save(pt_trace_path)

    data_set_path = os.path.join(out_dir, 'test_data_set_0')
    seq_id = 0
    while is_on_memory and os.path.exists(data_set_path):
        seq_id += 1
        data_set_path = os.path.join(
            out_dir, 'test_data_set_{:d}'.format(seq_id))
    os.makedirs(data_set_path, exist_ok=True)
    for i, (arg, name) in enumerate(zip(args, input_names)):
        f = os.path.join(data_set_path, 'input_{}.pb'.format(i))
        write_to_pb(f, arg, name)

    output_names = kwargs.get('output_names')
    if output_names is None:
        if isinstance(outs, dict):
            output_names = outs.keys()
        else:
            output_names = [None] * len(outs)
    for i, name in enumerate(output_names):
        if isinstance(outs, dict):
            out = outs[name]
        else:
            out = outs[i]
        if isinstance(out, (list, tuple)):
            assert len(out) == 1, \
                'Models returning nested lists/tuples are not supported yet'
            out = out[0]
        f = os.path.join(data_set_path, 'output_{}.pb'.format(i))
        write_to_pb(f, out, name)

    if output_grad is not False:
        if isinstance(output_grad, bool):
            output_grad = [torch.ones_like(outs[idx])
                           for idx in range(len(output_names))]
        if isinstance(output_grad, torch.Tensor):
            output_grad = [output_grad]
        for idx in range(len(output_names)):
            write_to_pb(
                os.path.join(data_set_path, 'gradient_input_{}.pb'.format(
                    idx)), output_grad[idx],
                output_names[idx])
        if len(output_names) == len(outs):
            torch.autograd.backward(outs, grad_tensors=output_grad)
        else:
            assert len(
                output_names) == 1, 'Single output names is only supported'
            outs[0].backward(output_grad[0])

        for i, (name, param) in enumerate(model.named_parameters()):
            f = os.path.join(data_set_path, 'gradient_{}.pb'.format(i))
            # NOTE: name does not follow C identifier syntax rules,
            # like "fc1.bias", not cleanse for now
            if param.grad is None:
                warnings.warn(
                    'Parameter `{}` does not have gradient value'.format(name))
            else:
                write_to_pb(f, param.grad, name)

    if user_meta is None:
        user_meta = {}

    if metadata:
        with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
            json.dump(_export_meta(model, out_dir, strip_large_tensor_data,
                                   user_meta), f, indent=2)
    elif user_meta:
        warnings.warn(
            '"user_meta" is given but "metadata" is False. '
            '"user_meta" is not exported.',
            UserWarning)

    if return_output:
        return outs
