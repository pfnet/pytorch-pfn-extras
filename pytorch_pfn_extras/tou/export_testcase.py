import datetime
import io
import json
import os
import subprocess
import warnings
from functools import reduce
import operator

import numpy
import onnx
import onnx.external_data_helper
import onnx.numpy_helper
import torch
import torch.autograd
from torch.onnx import OperatorExportTypes
from torch.onnx.symbolic_helper import _default_onnx_opset_version
from torch.onnx.utils import _export

from pytorch_pfn_extras.tou.annotate import init_annotate


LARGE_TENSOR_DATA_THRESHOLD = 100


def _export_meta(model, out_dir, strip_large_tensor_data):
    ret = {
        'generated_at': datetime.datetime.now().isoformat(),
        'output_directory': out_dir,
        'exporter': 'torch-onnx-utils',
        'strip_large_tensor_data': strip_large_tensor_data,
    }
    try:
        git_status = subprocess.Popen(['git', 'status'],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
        git_status.communicate()
        if git_status.returncode == os.EX_OK:
            def strip_cmd(cmd):
                return os.popen(cmd).read().strip()
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

    Copied from torch.onnx.utils.export
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

    return _export(model, args, f, _retain_param_name=True, **kwargs)


def is_large_tensor(tensor, threshold):
    size = reduce(operator.mul, tensor.dims, 1)
    return size > threshold


def _strip_raw_data(tensor):
    arr = onnx.numpy_helper.to_array(tensor)
    meta_dict = {}
    meta_dict['type'] = "stripped"
    meta_dict['average'] = float(numpy.average(arr))
    meta_dict['variance'] = float(numpy.var(arr))
    onnx.external_data_helper.set_external_data(
            tensor,
            location=json.dumps(meta_dict),
            length=len(tensor.raw_data))
    tensor.data_location = onnx.TensorProto.EXTERNAL
    tensor.ClearField('raw_data')
    return tensor


def _strip_large_initializer_raw_data(onnx_path, large_tensor_threshold):
    model = onnx.load(onnx_path)
    for init in model.graph.initializer:
        if is_large_tensor(init, large_tensor_threshold):
            _strip_raw_data(init)

    # Overwrite onnx file
    with open(onnx_path, "wb") as model_file:
        model_file.write(model.SerializeToString())


def export_testcase(
        model, args, out_dir, output_grad=False, metadata=True,
        model_overwrite=True, strip_large_tensor_data=False,
        large_tensor_threshold=LARGE_TENSOR_DATA_THRESHOLD, **kwargs):
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
    """

    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, 'model.onnx')
    is_on_memory = False
    if (not model_overwrite) and os.path.isfile(filename):
        filename = io.BytesIO(b"")
        is_on_memory = True
    model.zero_grad()
    opset_ver = kwargs.get('opset_version', None)
    if opset_ver is None:
        opset_ver = _default_onnx_opset_version
    with init_annotate(opset_ver):
        outs = _export_util(model, args, filename, **kwargs)

    if strip_large_tensor_data:
        _strip_large_initializer_raw_data(filename, large_tensor_threshold)

    def write_to_pb(f, tensor, name=None):
        array = tensor.detach().cpu().numpy()
        with open(f, 'wb') as fp:
            t = onnx.numpy_helper.from_array(array, name)
            if strip_large_tensor_data and is_large_tensor(
                    t, large_tensor_threshold):
                _strip_raw_data(t)
            fp.write(t.SerializeToString())

    if isinstance(args, torch.Tensor):
        args = args,
    if isinstance(outs, torch.Tensor):
        outs = outs,
    data_set_path = os.path.join(out_dir, 'test_data_set_0')
    seq_id = 0
    while is_on_memory and os.path.exists(data_set_path):
        seq_id += 1
        data_set_path = os.path.join(
            out_dir, 'test_data_set_{:d}'.format(seq_id))
    os.makedirs(data_set_path, exist_ok=True)
    for i, (arg, name) in enumerate(
            zip(args, kwargs.get('input_names', [None]*len(args)))):
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
            output_grad = \
                [torch.ones_like(outs[idx])
                 for idx in range(len(output_names))]
        if isinstance(output_grad, torch.Tensor):
            output_grad = [output_grad]
        for idx in range(len(output_names)):
            write_to_pb(
                os.path.join(data_set_path,
                             'gradient_input_{}.pb'.format(idx)),
                output_grad[idx],
                output_names[idx])
        if len(output_names) == len(outs):
            torch.autograd.backward(outs, grad_tensors=output_grad)
        else:
            assert len(output_names) == 1, \
                   'Single output names is only supported'
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

    if metadata:
        with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
            json.dump(_export_meta(model, out_dir, strip_large_tensor_data),
                      f, indent=2)
