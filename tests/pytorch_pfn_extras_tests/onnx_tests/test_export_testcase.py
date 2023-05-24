import io
import os
import json
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper
import pytest
import pytorch_pfn_extras
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pfn_extras.onnx import export
from pytorch_pfn_extras.onnx import export_testcase
from pytorch_pfn_extras.onnx import is_large_tensor
from pytorch_pfn_extras.onnx import LARGE_TENSOR_DATA_THRESHOLD
import pytorch_pfn_extras.onnx._constants
from pytorch_pfn_extras.onnx.strip_large_tensor import _strip_large_tensor_tool_impl
from pytorch_pfn_extras.onnx.unstrip_tensor import unstrip
from pytorch_pfn_extras.onnx.pfto_exporter.torch_reconstruct import reconstruct


output_dir = 'out'
output_counter = {}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _get_output_dir(d, **kwargs):
    output_dir_base = 'out'
    opset_ver = kwargs.get('opset_version', pytorch_pfn_extras.onnx._constants.onnx_default_opset)

    test_name = os.getenv("PYTEST_CURRENT_TEST").split(':')[-1].split(' ')[0]
    if d:
        test_name = f"{test_name}_{d}"
    if "model_overwrite" not in kwargs:
        if test_name in output_counter:
            output_counter[test_name] += 1
            test_name = f"{test_name}_{output_counter[test_name]}"
        else:
            output_counter[test_name] = 0

    output_dir = os.path.join(
        output_dir_base, 'opset{}'.format(opset_ver), test_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _helper(model, args, d, use_pfto=True, check_reconstruct=True, **kwargs):
    output_dir = _get_output_dir(d, **kwargs)
    if 'training' not in kwargs:
        kwargs['training'] = model.training
    if 'do_constant_folding' not in kwargs:
        kwargs['do_constant_folding'] = False
    if 'metadata' not in kwargs:
        kwargs["metadata"] = False
    if "strip_doc_string" not in kwargs:
        kwargs["strip_doc_string"] = False
    export_testcase(model, args, output_dir, use_pfto=use_pfto, **kwargs)
    if check_reconstruct and use_pfto and not kwargs["strip_doc_string"]:
        reconstruct(pytorch_pfn_extras.onnx.load_model(
            os.path.join(output_dir, "model.onnx")))
    return output_dir


@pytest.mark.filterwarnings("ignore:Named tensors .* experimental:UserWarning")
def test_export_testcase():
    model = Net().to('cpu')
    x = torch.zeros((1, 1, 28, 28))

    output_dir = _helper(
        model, x, 'mnist', output_grad=True, metadata=True,
        check_reconstruct=True, verbose=False)

    assert os.path.isdir(output_dir)
    assert os.path.isfile(os.path.join(output_dir, 'meta.json'))
    assert os.path.isfile(os.path.join(output_dir, 'model.onnx'))
    test_data_set_dir = os.path.join(output_dir, 'test_data_set_0')
    expected_input_0_path = os.path.join(test_data_set_dir, 'input_0.pb')
    assert os.path.isfile(expected_input_0_path)
    assert _to_array(expected_input_0_path).shape == (1, 1, 28, 28)
    assert os.path.isfile(os.path.join(test_data_set_dir, 'output_0.pb'))
    assert os.path.isfile(os.path.join(
        test_data_set_dir, 'gradient_input_0.pb'))

    for i in range(8):
        assert os.path.isfile(os.path.join(
            test_data_set_dir, 'gradient_{}.pb'.format(i)))
    assert not os.path.isfile(os.path.join(test_data_set_dir, 'gradient_8.pb'))


def test_export_filename():
    model = nn.Sequential(nn.Linear(5, 10, bias=False))
    x = torch.zeros((2, 5))

    output_dir = _get_output_dir('export_filename')
    model_path = os.path.join(output_dir, 'model.onnx')

    with pytest.warns(UserWarning):
        out = export(model, x, model_path, return_output=True)

    assert os.path.isfile(model_path)
    expected_out = torch.zeros((2, 10))  # check only shape size
    np.testing.assert_allclose(
        out.detach().cpu().numpy(), expected_out.detach().cpu().numpy())


def test_export_testcase_return_output():
    model = nn.Sequential(nn.Linear(5, 10, bias=False))
    x = torch.zeros((2, 5))

    output_dir = _get_output_dir('export_filename')

    if pytorch_pfn_extras.requires("1.6.0") and not pytorch_pfn_extras.requires("1.13"):
        with pytest.warns(UserWarning):
            (out,) = export_testcase(model, x, output_dir, return_output=True)
    else:
        (out,) = export_testcase(model, x, output_dir, return_output=True)

    assert os.path.isfile(os.path.join(output_dir, 'model.onnx'))
    expected_out = torch.zeros((2, 10))  # check only shape size
    np.testing.assert_allclose(
        out.detach().cpu().numpy(), expected_out.detach().cpu().numpy())


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_export_stream():
    model = nn.Sequential(nn.Linear(5, 10, bias=False))
    x = torch.zeros((2, 5))

    bytesio = io.BytesIO()
    assert len(bytesio.getvalue()) == 0
    out = export(model, x, bytesio, return_output=True)

    assert len(bytesio.getvalue()) > 0
    expected_out = torch.zeros((2, 10))  # check only shape size
    np.testing.assert_allclose(
        out.detach().cpu().numpy(), expected_out.detach().cpu().numpy())


def test_cuda_tensor():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')

    device = 'cuda'
    model = Net().to(device)
    x = torch.zeros((1, 1, 28, 28), device=device)

    _helper(model, x, 'mnist_cuda', output_grad=True)


def test_model_not_overwrite():
    model = Net().to('cpu')
    x = torch.zeros((1, 1, 28, 28))

    dir_name = 'multiple_test_dataset'
    output_dir = _helper(model, x, dir_name)
    assert os.path.isdir(output_dir)

    output_dir = _helper(
        model, x + 0.5, dir_name,
        model_overwrite=False)

    test_data_set_dir = os.path.join(output_dir, 'test_data_set_1')
    assert os.path.isfile(os.path.join(test_data_set_dir, 'input_0.pb'))
    assert os.path.isfile(os.path.join(test_data_set_dir, 'output_0.pb'))


def _to_array(f, name=None):
    assert os.path.isfile(f)
    onnx_tensor = onnx.TensorProto()
    with open(f, 'rb') as fp:
        onnx_tensor.ParseFromString(fp.read())
    if name is not None:
        assert onnx_tensor.name == name
    return onnx.numpy_helper.to_array(onnx_tensor)


def test_backward():
    model = nn.Sequential(nn.Linear(5, 10, bias=False))
    x = torch.ones((2, 5))

    output_dir = _helper(model, x, 'backword_default', output_grad=True)

    assert os.path.isdir(output_dir)
    test_data_set_dir = os.path.join(output_dir, 'test_data_set_0')
    assert os.path.isdir(test_data_set_dir)

    grad = _to_array(os.path.join(test_data_set_dir, 'gradient_0.pb'))
    expected_grad = np.full((10, 5), 2.0, dtype=np.float32)
    np.testing.assert_allclose(grad, expected_grad)


def test_backward_custom_input():
    model = nn.Sequential(nn.Linear(5, 10, bias=False))
    x = torch.ones((2, 5))
    grad_in = torch.ones((2, 10)) * 0.5

    output_dir = _helper(
        model, x, 'backword_custom_input', output_grad=grad_in,
        output_names=['output0'])

    assert os.path.isdir(output_dir)
    test_data_set_dir = os.path.join(output_dir, 'test_data_set_0')
    assert os.path.isdir(test_data_set_dir)

    output_grad_in = _to_array(
        os.path.join(test_data_set_dir, 'gradient_input_0.pb'), 'output0')
    np.testing.assert_allclose(output_grad_in, grad_in)

    grad = _to_array(os.path.join(test_data_set_dir, 'gradient_0.pb'))
    expected_grad = np.full((10, 5), 1.0, dtype=np.float32)
    np.testing.assert_allclose(grad, expected_grad)


@pytest.mark.filterwarnings(
    "ignore::torch.jit.TracerWarning", "ignore::UserWarning")
def test_backward_multiple_input():
    model = nn.GRU(input_size=10, hidden_size=3, num_layers=1)
    input = torch.ones((4, 5, 10), requires_grad=True)
    h = torch.ones((1, 5, 3), requires_grad=True)

    grads = [torch.ones((4, 5, 3)) / 2, torch.ones((1, 5, 3)) / 3]
    output_dir = _helper(model, (input, h), 'backward_multiple_input',
                         output_grad=grads,
                         output_names=['output0', 'output1'])
    assert os.path.isdir(output_dir)
    test_data_set_dir = os.path.join(output_dir, 'test_data_set_0')
    assert os.path.isdir(test_data_set_dir)

    model.zero_grad()
    exp_out1, exp_out2 = model.forward(input, h)
    torch.autograd.backward(
        tensors=[exp_out1, exp_out2],
        grad_tensors=grads)

    output1_grad_in = _to_array(
        os.path.join(test_data_set_dir, 'gradient_input_0.pb'), 'output0')
    np.testing.assert_allclose(grads[0], output1_grad_in)
    output2_grad_in = _to_array(
        os.path.join(test_data_set_dir, 'gradient_input_1.pb'), 'output1')
    np.testing.assert_allclose(grads[1], output2_grad_in)

    for i, (name, param) in enumerate(model.named_parameters()):
        actual_grad = _to_array(
            os.path.join(test_data_set_dir, 'gradient_{}.pb'.format(i)), name)
        np.testing.assert_allclose(param.grad, actual_grad)


@pytest.mark.filterwarnings(
    "ignore::torch.jit.TracerWarning", "ignore::UserWarning")
def test_export_testcase_strip_large_tensor_data():
    if not pytorch_pfn_extras.requires("1.6.0"):
        pytest.skip('skip for PyTorch 1.5 or earlier')

    model = Net().to('cpu')
    x = torch.zeros((1, 1, 28, 28))

    output_dir = _helper(
        model, x, 'mnist_stripped_tensor_data',
        output_grad=True, strip_large_tensor_data=True,
        metadata=True)

    assert os.path.isdir(output_dir)
    assert os.path.isfile(os.path.join(output_dir, 'meta.json'))
    assert os.path.isfile(os.path.join(output_dir, 'model.onnx'))
    test_data_set_dir = os.path.join(output_dir, 'test_data_set_0')
    assert os.path.isfile(os.path.join(test_data_set_dir, 'input_0.pb'))
    assert os.path.isfile(os.path.join(test_data_set_dir, 'output_0.pb'))

    for i in range(8):
        assert os.path.isfile(os.path.join(
            test_data_set_dir, 'gradient_{}.pb'.format(i)))
    assert not os.path.isfile(os.path.join(test_data_set_dir, 'gradient_8.pb'))

    with open(os.path.join(output_dir, 'meta.json')) as metaf:
        metaj = json.load(metaf)
        assert metaj['strip_large_tensor_data']

    def is_stripped_with_check(tensor):
        if is_large_tensor(tensor, LARGE_TENSOR_DATA_THRESHOLD):
            assert tensor.data_location == onnx.TensorProto.EXTERNAL
            assert tensor.external_data[0].key == 'location'
            meta = json.loads(tensor.external_data[0].value)
            assert meta['type'] == 'stripped'
            assert type(meta['average']) == float
            assert type(meta['variance']) == float
            return True
        assert len(tensor.external_data) == 0
        return False

    onnx_model = onnx.load(os.path.join(
        output_dir, 'model.onnx'), load_external_data=False)

    check_stripped = [
        is_stripped_with_check(init) for init in onnx_model.graph.initializer]
    # this testcase tests strip, so output mode is no stripped, test is failed
    assert any(check_stripped)

    for pb_filepath in ('input_0.pb', 'output_0.pb'):
        with open(os.path.join(test_data_set_dir, pb_filepath), 'rb') as f:
            tensor = onnx.TensorProto()
            tensor.ParseFromString(f.read())
            is_stripped_with_check(tensor)

    # check re-load stripped onnx
    _strip_large_tensor_tool_impl(
        os.path.join(output_dir, 'model.onnx'),
        os.path.join(output_dir, 'model_re.onnx'),
        LARGE_TENSOR_DATA_THRESHOLD)
    assert os.path.isfile(os.path.join(output_dir, 'model_re.onnx'))
    # loading check
    onnx.load(
        os.path.join(output_dir, 'model_re.onnx'), load_external_data=False)

    # unstrip test
    unstrip_output_dir = _get_output_dir('mnist_unstripped_tensor_data')
    unstrip(output_dir, out_path=unstrip_output_dir)

    def is_unstripped_with_check(tensor):
        if is_large_tensor(tensor, LARGE_TENSOR_DATA_THRESHOLD):
            assert tensor.data_location == onnx.TensorProto.DEFAULT
            return True
        return False

    onnx_paths = Path(unstrip_output_dir).glob('*.onnx')
    check_unstripped = []
    for onnx_path in onnx_paths:
        onnx_model = onnx.load(onnx_path, load_external_data=False)
        check_unstripped.extend([
            is_unstripped_with_check(init) for init in onnx_model.graph.initializer])
    assert len(check_unstripped) > 0
    assert any(check_unstripped)

    pb_paths = Path(unstrip_output_dir).glob('**/*.pb')
    checked = False
    for pb_path in pb_paths:
        tensor = onnx.TensorProto()
        with open(pb_path, 'rb') as f:
            tensor.ParseFromString(f.read())
        checked = is_unstripped_with_check(tensor) or checked
    assert checked, 'more than one data is unstripped'

    with open(os.path.join(unstrip_output_dir, 'meta.json')) as metaf:
        metaj = json.load(metaf)
        assert not metaj['strip_large_tensor_data']


def test_export_testcase_options():
    model = Net().to('cpu')
    x = torch.zeros((1, 1, 28, 28))

    output_dir = _helper(
        model, x, 'mnist_stripped_tensor_data',
        opset_version=11, strip_doc_string=False,
        check_reconstruct=False)

    onnx_model = onnx.load(os.path.join(
        output_dir, 'model.onnx'), load_external_data=False)
    assert onnx_model.opset_import[0].version == 11
    assert onnx_model.graph.node[0].doc_string != ''


class NetWithUnusedInput(nn.Module):
    def __init__(self):
        super(NetWithUnusedInput, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x, unused):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


@pytest.mark.filterwarnings("ignore:Unused input:UserWarning")
@pytest.mark.parametrize("keep_initializers_as_inputs", [None, True, False])
def test_export_testcase_with_unused_input(keep_initializers_as_inputs):
    if not pytorch_pfn_extras.requires("1.7.0"):
        pytest.skip('skip for PyTorch 1.6 or earlier')

    model = NetWithUnusedInput().to('cpu')
    x = torch.zeros((1, 1, 28, 28))
    unused = torch.zeros((1,))

    # Without input_names
    output_dir = _helper(
        model, args=(x, unused), d='net_with_unused_input_without_input_names',
        opset_version=11, strip_doc_string=False,
        keep_initializers_as_inputs=keep_initializers_as_inputs)
    assert os.path.isdir(output_dir)
    test_data_set_dir = os.path.join(output_dir, 'test_data_set_0')
    assert os.path.exists(os.path.join(test_data_set_dir, 'input_0.pb'))
    assert not os.path.exists(os.path.join(test_data_set_dir, 'input_1.pb'))

    xmodel = onnx.load_model(os.path.join(output_dir, 'model.onnx'))
    assert xmodel.graph.input[0].name == 'input_0'
    assert len(xmodel.graph.input) == 1 or \
        xmodel.graph.input[1].name != 'input_1'

    # With input_names
    output_dir = _helper(
        model, args=(x, unused), d='net_with_unused_input_with_input_names',
        opset_version=11, strip_doc_string=False,
        keep_initializers_as_inputs=keep_initializers_as_inputs,
        input_names=['x', 'unused'])
    assert os.path.isdir(output_dir)
    test_data_set_dir = os.path.join(output_dir, 'test_data_set_0')
    assert os.path.exists(os.path.join(test_data_set_dir, 'input_0.pb'))
    assert not os.path.exists(os.path.join(test_data_set_dir, 'input_1.pb'))

    xmodel = onnx.load_model(os.path.join(output_dir, 'model.onnx'))
    assert xmodel.graph.input[0].name == 'x'
    assert len(xmodel.graph.input) == 1 or \
        xmodel.graph.input[1].name != 'unused'


def test_user_meta():
    model = nn.Sequential(nn.Linear(5, 10, bias=False))
    x = torch.ones((2, 5))

    output_dir = _helper(model, x, 'meta_without_user_meta', metadata=True)
    with open(os.path.join(output_dir, "meta.json")) as metaf:
        assert "user_meta" not in json.load(metaf)

    output_dir = _helper(model, x, 'meta_with_user_meta', metadata=True,
                         user_meta={"user_key": "user_value"})
    with open(os.path.join(output_dir, "meta.json")) as metaf:
        assert json.load(metaf)["user_meta"]["user_key"] == "user_value"

    with pytest.warns(UserWarning):
        output_dir = _helper(model, x, 'without_meta_with_user_meta',
                             metadata=False,
                             user_meta={"user_key": "user_value"})
        assert not os.path.exists(os.path.join(output_dir, 'meta.json'))


def test_export_pt():
    model = nn.Sequential(nn.Linear(5, 10, bias=False))
    x = torch.ones((2, 5))

    output_dir = _helper(model, x, 'export_script_pt',
                         export_torch_script=True,
                         export_torch_trace=False)
    assert os.path.exists(os.path.join(output_dir, 'model_script.pt'))
    assert not os.path.exists(os.path.join(output_dir, 'model_trace.pt'))

    output_dir = _helper(model, x, 'export_trace_pt',
                         export_torch_script=False,
                         export_torch_trace=True)
    assert not os.path.exists(os.path.join(output_dir, 'model_script.pt'))
    assert os.path.exists(os.path.join(output_dir, 'model_trace.pt'))


@pytest.mark.skipif(
    not pytorch_pfn_extras.requires("1.10.0"),
    reason='skip for PyTorch 1.9 or earlier')
def test_export_scripted():

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Linear(5, 10, bias=False)

        def forward(self, x, b):
            if b:  # IF statement to check scripted (not traced) IR
                x = -x
            x = self.linear(x)
            return x

    model = torch.jit.script(Net())
    x = torch.zeros((2, 5))
    b = torch.tensor(True)

    output_dir = _get_output_dir('export_scripted')
    (out,) = export_testcase(
        model, (x, b), output_dir, return_output=True, training=model.training,
        do_constant_folding=False)

    assert os.path.isfile(os.path.join(output_dir, 'model.onnx'))
    expected_out = torch.zeros((2, 10))  # check only shape size
    np.testing.assert_allclose(
        out.detach().cpu().numpy(), expected_out.detach().cpu().numpy())
