from collections.abc import Callable
import datetime
import glob
import io
import itertools
import json
import os
import subprocess
from typing import Any, Dict, IO, Mapping, Optional, Sequence, Tuple, Union
import warnings

import onnx
import onnx.numpy_helper
import pytorch_pfn_extras
import pytorch_pfn_extras.onnx._constants
import torch
import torch.autograd
from torch.onnx import OperatorExportTypes
from torch.onnx.utils import \
    _export as torch_export, _model_to_graph as torch_model_to_graph, _decide_input_format

from pytorch_pfn_extras.onnx import _as_output as as_output
from pytorch_pfn_extras.onnx import _grad as grad
from pytorch_pfn_extras.onnx import _lax as lax
from pytorch_pfn_extras.onnx.annotate import init_annotate
from pytorch_pfn_extras.onnx.strip_large_tensor import \
    LARGE_TENSOR_DATA_THRESHOLD
from pytorch_pfn_extras.onnx.strip_large_tensor import is_large_tensor
from pytorch_pfn_extras.onnx.strip_large_tensor import _strip_raw_data
from pytorch_pfn_extras.onnx.strip_large_tensor import \
    _strip_large_initializer_raw_data
from pytorch_pfn_extras.onnx.pfto_exporter.export import export as pfto_export


def _model_to_graph_with_value_names(
        *args: Any,
        add_value_names: bool = True,
        **kwargs: Any,
) -> Tuple[torch._C.Graph, Dict[str, Any], Any]:
    g, p, o = torch_model_to_graph(*args, **kwargs)  # type: ignore[no-untyped-call]
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


def _export_meta(
        model: torch.nn.Module,
        out_dir: str,
        strip_large_tensor_data: bool,
        user_meta: Mapping[str, Any],
) -> Dict[str, Any]:
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

        def strip_cmd(cmd: str) -> str:
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


def _export_util(
        model: torch.nn.Module,
        args: Sequence[Any],
        f: IO,
        return_output: bool = False,
        custom_exporter: Optional[Callable] = None,
        **kwargs: Any,
) -> Any:
    """Wrap operator type to export

    Copied from torch.onnx.utils.export, to get output values.
    """
    aten = kwargs.get('aten', False)
    export_raw_ir = kwargs.get('export_raw_ir', False)
    operator_export_type = kwargs.get('operator_export_type', None)

    if aten or export_raw_ir:
        assert operator_export_type is None
        assert aten ^ export_raw_ir
        operator_export_type = OperatorExportTypes.ONNX_ATEN
    elif operator_export_type is None:
        if pytorch_pfn_extras.requires("2.5.0"):
            use_onnx_aten_fallback = False
        else:
            use_onnx_aten_fallback = torch.onnx._CAFFE2_ATEN_FALLBACK  # type: ignore[attr-defined]
        if use_onnx_aten_fallback:
            operator_export_type = OperatorExportTypes.ONNX_ATEN_FALLBACK
        else:
            operator_export_type = OperatorExportTypes.ONNX

    old_model_to_graph = torch.onnx.utils._model_to_graph
    # TODO(ecastill) _model_to_graph shouldn't be direclty overriden
    # This is a temporal workaround until a fix is introduced in PyTorch.
    try:
        torch.onnx.utils._model_to_graph = _model_to_graph_with_value_names
        checker_error = getattr(torch.onnx, "CheckerError", None)
        if checker_error is None:
            checker_error = getattr(torch.onnx.utils, "ONNXCheckerError", None)  # type: ignore[attr-defined]
        if checker_error is None:
            # PyTorch 2.6 does not have either of exception classes above.
            # As check by onnx.checker has been removed we no longer need to
            # capture the exception.
            class checker_error(RuntimeError):  # type: ignore[no-redef]
                pass
        try:
            enable_onnx_checker = kwargs.pop('enable_onnx_checker', None)
            if pytorch_pfn_extras.requires("2.5.0") and enable_onnx_checker:
                warnings.warn("onnx checker not supported from 2.5", UserWarning)
            if custom_exporter is None:
                return torch_export(  # type: ignore[no-untyped-call]
                    model, args, f, **kwargs)
            else:
                return custom_exporter(  # type: ignore[no-untyped-call]
                    model, args, f, **kwargs)
        except checker_error:  # type: ignore[misc]
            if enable_onnx_checker:
                raise
            if return_output:
                # Re-run the model to obtain the output.
                return model(*args)
    finally:
        torch.onnx.utils._model_to_graph = old_model_to_graph


def _export(
        model: torch.nn.Module,
        args: Sequence[Any],
        strip_large_tensor_data: bool = False,
        large_tensor_threshold: int = LARGE_TENSOR_DATA_THRESHOLD,
        use_pfto: bool = False,
        return_output: bool = True,
        chrome_tracing: str = "",
        **kwargs: Any,
) -> Tuple[onnx.ModelProto, Any]:
    model.zero_grad()
    bytesio = io.BytesIO()
    opset_ver = kwargs.get('opset_version', None)
    force_verbose = False

    if "training" in kwargs and (isinstance(kwargs["training"], bool) or kwargs['training'] is None):
        kwargs["training"] = torch.onnx.TrainingMode.TRAINING \
            if kwargs["training"] \
            else torch.onnx.TrainingMode.EVAL

    if opset_ver is None:
        opset_ver = pytorch_pfn_extras.onnx._constants.onnx_default_opset
        kwargs['opset_version'] = opset_ver
    if use_pfto:
        strip_doc_string = kwargs.get('strip_doc_string', True)
        kwargs['strip_doc_string'] = False
    else:
        strip_doc_string = kwargs.pop('strip_doc_string', True)
        if (not kwargs.get('verbose', False) and
                not pytorch_pfn_extras.requires("2.6.0")):
            # torch.onnx.log was removed in PyTorch 2.6.0.
            # https://github.com/pytorch/pytorch/pull/133825
            force_verbose = True
            original_log = torch.onnx.log  # type: ignore[attr-defined]
            #  Following line won't work because verbose mode always
            # enable logging so we are replacing python function instead:
            # torch.onnx.disable_log()
            def no_op(*args: Any) -> None:
                pass

            torch.onnx.log = no_op  # type: ignore[attr-defined]
        kwargs['verbose'] = True
    # Exted args with kwargs (including default values)
    args = _decide_input_format(model, args)  # type: ignore[no-untyped-call]
    with init_annotate(model, opset_ver) as ann, \
            as_output.trace(model) as (model, outputs), \
            grad.init_grad_state(), \
            lax.init_lax_state():
        if use_pfto:
            outs = pfto_export(
                model, args, bytesio, chrome_tracing=chrome_tracing, **kwargs)
        else:
            if chrome_tracing:
                with torch.profiler.profile() as prof:
                    outs = _export_util(
                        model, args, bytesio, return_output=return_output, **kwargs)
                prof.export_chrome_trace(chrome_tracing)
            else:
                outs = _export_util(
                    model, args, bytesio, return_output=return_output, **kwargs)
        onnx_graph = onnx.load(io.BytesIO(bytesio.getvalue()))
        onnx_graph = ann.set_annotate(onnx_graph)
        onnx_graph = ann.reorg_anchor(onnx_graph)
        outputs.add_outputs_to_model(onnx_graph)
        lax.postprocess(onnx_graph)
        if strip_doc_string:
            for node in onnx_graph.graph.node:
                node.doc_string = b''

    if strip_large_tensor_data:
        _strip_large_initializer_raw_data(onnx_graph, large_tensor_threshold)

    if force_verbose:
        # torch.onnx.enable_log()
        torch.onnx.log = original_log  # type: ignore[attr-defined]

    return onnx_graph, outs


def export(
        model: torch.nn.Module,
        args: Sequence[Any],
        f: IO,
        return_output: bool = False,
        strip_large_tensor_data: bool = False,
        large_tensor_threshold: int = LARGE_TENSOR_DATA_THRESHOLD,
        chrome_tracing: str = "",
        **kwargs: Any,
) -> Any:
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
        return_output=return_output,
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
        model: Union[torch.nn.Module, torch.jit.ScriptModule],
        args: Any,
        out_dir: str,
        *,
        output_grad: Union[bool, torch.Tensor, Sequence[torch.Tensor]] = False,
        metadata: bool = True,
        model_overwrite: bool = True,
        strip_large_tensor_data: bool = False,
        large_tensor_threshold: int = LARGE_TENSOR_DATA_THRESHOLD,
        return_output: bool = False,
        user_meta: Optional[Mapping[str, Any]] = None,
        export_torch_script: bool = False,
        export_torch_trace: bool = False,
        export_chrome_tracing: bool = True,
        **kwargs: Any,
) -> Any:
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

    .. note:: 
        When exporting a model whose forward takes keyword arguments of ``torch.Tensor`` type,
        you can pass them by putting a dict as the last element of ``args``.
        When the keyword arguments have default values, you need to explicitly include
        them into the dict.
        Also, you must explicitly specify ``input_names`` that are the names of both positional
        and keyword arguments.
    """

    training_mode = kwargs.get("training", torch.onnx.TrainingMode.EVAL)

    if (not isinstance(output_grad, bool) or output_grad) and training_mode is torch.onnx.TrainingMode.EVAL:
        warnings.warn(
            "You are exporting testcase with gradients but `training` is set to "
            "torch.onnx.TrainingMode.EVAL. This will constant fold your parameters "
            "and affect the backpropagation. Please set the `training` to "
            "torch.onnx.TrainingMode.TRAINING."
        )

    if user_meta is None:
        user_meta = {}

    chrome_tracing = ""
    if export_chrome_tracing:
        chrome_tracing = os.path.join(out_dir, "export_trace.json.gz")
        if os.path.exists(chrome_tracing):
            os.remove(chrome_tracing)

    os.makedirs(out_dir, exist_ok=True)
    if isinstance(args, torch.Tensor):
        args = args,
        
    def has_kwargs_in_args(args: tuple) -> bool:
        return len(args) >= 1 and isinstance(args[-1], dict)

    if has_kwargs_in_args(args):
        assert "input_names" in kwargs, 'export_testcase needs explicit "input_names" when exporting with kwargs'
        named_args_list = list(args[:-1])
        for key, tensor in args[-1].items():
            assert isinstance(tensor, torch.Tensor)
            named_args_list.append(tensor)
        named_args = tuple(named_args_list)
    else:
        named_args = args

    input_names = kwargs.pop(
        'input_names',
        ['input_{}'.format(i) for i in range(len(args))])
    assert len(input_names) == len(named_args)
    assert not isinstance(args, torch.Tensor)

    onnx_graph, outs = _export(
        model, args, strip_large_tensor_data, large_tensor_threshold,
        input_names=input_names, chrome_tracing=chrome_tracing, **kwargs)
    if isinstance(model, torch.jit.ScriptModule):
        assert outs is None
        outs = model(*args)
    if isinstance(outs, torch.Tensor):
        outs = outs,
    assert outs is not None
    outs = torch._C._jit_flatten(outs)[0]
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
    args = [named_args[i] for i in used_input_index_list]

    output_path = os.path.join(out_dir, 'model.onnx')
    is_on_memory = True
    if model_overwrite or (not os.path.isfile(output_path)):
        is_on_memory = False
        with open(output_path, 'wb') as fp:
            fp.write(onnx_graph.SerializeToString())

    def write_to_pb(f: str, tensor: torch.Tensor, name: Optional[str] = None) -> None:
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
            torch.jit.script(  # type: ignore[no-untyped-call]
                model).save(pt_script_path)

    if export_torch_trace:
        pt_trace_path = os.path.join(out_dir, 'model_trace.pt')
        if model_overwrite or (not os.path.isfile(pt_trace_path)):
            torch.jit.trace(  # type: ignore[no-untyped-call]
                model, args).save(pt_trace_path)

    data_set_path = os.path.join(out_dir, 'test_data_set_0')
    seq_id = 0
    while is_on_memory and os.path.exists(data_set_path):
        seq_id += 1
        data_set_path = os.path.join(
            out_dir, 'test_data_set_{:d}'.format(seq_id))
    os.makedirs(data_set_path, exist_ok=True)
    for pb_name in glob.glob(os.path.join(data_set_path, "*.pb")):
        os.remove(pb_name)
    flat_inputs = torch._C._jit_flatten(named_args)[0]
    for i, (arg, name) in enumerate(zip(flat_inputs, input_names)):
        f = os.path.join(data_set_path, 'input_{}.pb'.format(i))
        write_to_pb(f, arg, name)

    output_names: Optional[Sequence[Optional[str]]]
    output_names = kwargs.get('output_names')  # type: ignore[assignment]
    if output_names is None:
        if isinstance(outs, dict):
            output_names = list(outs.keys())
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

    if metadata:
        with open(os.path.join(out_dir, 'meta.json'), 'w') as fp_text:
            json.dump(_export_meta(model, out_dir, strip_large_tensor_data,
                                   user_meta), fp_text, indent=2)
    elif user_meta:
        warnings.warn(
            '"user_meta" is given but "metadata" is False. '
            '"user_meta" is not exported.',
            UserWarning)

    if not metadata and strip_large_tensor_data:
        warnings.warn(
            '"strip_large_tensor_data" is given but "metadata" is False. '
            'It would be harder to determine whether testcase or model is stripped.',
            UserWarning)

    if return_output:
        return outs
