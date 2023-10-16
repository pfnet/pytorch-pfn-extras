import os
import sys

import onnx
import pytest
import pytorch_pfn_extras
import torch
import torch.nn as nn
import torch.onnx

from pytorch_pfn_extras.onnx import lax
from pytorch_pfn_extras_tests.onnx_tests.test_export_testcase import _helper, _ort_session


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_fori_loop_no_export():
    if not pytorch_pfn_extras.requires("1.8.0"):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    torch.manual_seed(0)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Linear(1, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x * 0.5
            h = lax.fori_loop(1, 4, lambda it, val: it * self.relu(self.linear(val)), x)
            return h + 1

    model = Net()
    x = torch.ones((1, 1))
    y = model(x)
    v = x * 0.5
    for i in range(1, 4):
        v = i * model.relu(model.linear(v))
    y_expected = v + 1
    torch.testing.assert_close(y, y_expected)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_fori_loop():
    if not pytorch_pfn_extras.requires('1.8.0'):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    if pytorch_pfn_extras.requires('1.10.0') and sys.platform == 'win32':
        pytest.skip('ONNX grad test does not work in windows CI for torch >= 1.10')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Linear(1, 1)
            self.linear.weight.data[:] = 2
            self.linear.bias.data[:] = 3

        def forward(self, x):
            h = lax.fori_loop(1, 4, lambda it, val: it * self.linear(val) + 0.5, x)
            return h

    model = Net()
    x = torch.tensor([[0], [1]]).float()
    output_dir = _helper(
        model,
        x,
        'fori_loop',
        input_names=("x",),
        enable_onnx_checker=False,
        use_pfto=False,
        do_constant_folding=False,
    )

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    assert len([x for x in actual_onnx.graph.node if x.op_type == "Loop"]) == 1
    ort_session = _ort_session(os.path.join(output_dir, "model.onnx"))
    actual = ort_session.run(None, {"x": x.cpu().numpy()})
    expected = model(x)
    torch.testing.assert_close(expected, torch.tensor(actual[0]))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_fori_loop_with_tuple_state():
    if not pytorch_pfn_extras.requires('1.8.0'):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    if pytorch_pfn_extras.requires('1.10.0') and sys.platform == 'win32':
        pytest.skip('ONNX grad test does not work in windows CI for torch >= 1.10')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Linear(1, 1)
            self.linear.weight.data[:] = 2
            self.linear.bias.data[:] = 3

        def forward(self, x):
            def body(it, val):
                h0, h1 = val
                return it * self.linear(h0), it * self.linear(h1) + 0.5

            h0, h1 = lax.fori_loop(1, 4, body, (x, x + 1))
            return h0 + h1

    model = Net()
    x = torch.tensor([[0], [1]]).float()
    output_dir = _helper(
        model,
        x,
        'fori_loop_tuple_state',
        input_names=("x",),
        enable_onnx_checker=False,
        use_pfto=False,
        do_constant_folding=False,
    )

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    assert len([x for x in actual_onnx.graph.node if x.op_type == "Loop"]) == 1
    ort_session = _ort_session(os.path.join(output_dir, "model.onnx"))
    actual = ort_session.run(None, {"x": x.cpu().numpy()})
    expected = model(x)
    torch.testing.assert_close(expected, torch.tensor(actual[0]))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_while_loop_no_export():
    if not pytorch_pfn_extras.requires('1.8.0'):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    if pytorch_pfn_extras.requires('1.10.0') and sys.platform == 'win32':
        pytest.skip('ONNX grad test does not work in windows CI for torch >= 1.10')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Linear(1, 1)
            self.linear.weight.data[:] = 2
            self.linear.bias.data[:] = 3

        def forward(self, x):
            def cond_fn(x):
                return x.sum() < 100

            def body_fn(x):
                return self.linear(x)
            h = lax.while_loop(cond_fn, body_fn, x)
            return h

    model = Net()
    x = torch.tensor([[0], [1]]).float()
    out = model(x)
    assert out.sum().item() > 100


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore:Converting a tensor to a Python boolean might cause the trace to be incorrect:torch.jit.TracerWarning")
def test_while_loop():
    if not pytorch_pfn_extras.requires('1.8.0'):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    if pytorch_pfn_extras.requires('1.10.0') and sys.platform == 'win32':
        pytest.skip('ONNX grad test does not work in windows CI for torch >= 1.10')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Linear(1, 1)
            self.linear.weight.data[:] = 2
            self.linear.bias.data[:] = 3

        def forward(self, x):
            def cond_fn(x):
                return x.sum() < 100

            def body_fn(x):
                return self.linear(x)
            h = lax.while_loop(cond_fn, body_fn, x)
            return h

    model = Net()
    x = torch.tensor([[0], [1]]).float()
    output_dir = _helper(
        model,
        x,
        'while_loop',
        input_names=("x",),
        enable_onnx_checker=False,
        use_pfto=False,
        do_constant_folding=False,
    )

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    assert len([x for x in actual_onnx.graph.node if x.op_type == "Loop"]) == 1
    ort_session = _ort_session(os.path.join(output_dir, "model.onnx"))
    actual = ort_session.run(None, {"x": x.cpu().numpy()})
    expected = model(x)
    torch.testing.assert_close(expected, torch.tensor(actual[0]))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_cond_no_export():
    if not pytorch_pfn_extras.requires('1.8.0'):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    if pytorch_pfn_extras.requires('1.10.0') and sys.platform == 'win32':
        pytest.skip('ONNX grad test does not work in windows CI for torch >= 1.10')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Linear(1, 1)
            self.linear.weight.data[:] = 2
            self.linear.bias.data[:] = 3

        def forward(self, x):
            def true_fn(x):
                return self.linear(x)

            def false_fn(x):
                return -x

            h = lax.cond(x.sum().long() % 2 == 0, true_fn, false_fn, x)
            return h

    model = Net()
    x = torch.tensor([[0], [1]]).float()
    out = model(x)
    assert out[0] == 0
    assert out[1] == -1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore:Converting a tensor to a Python boolean might cause the trace to be incorrect:torch.jit.TracerWarning")
def test_cond():
    if not pytorch_pfn_extras.requires('1.8.0'):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    if pytorch_pfn_extras.requires('1.10.0') and sys.platform == 'win32':
        pytest.skip('ONNX grad test does not work in windows CI for torch >= 1.10')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Linear(1, 1)
            self.linear.weight.data[:] = 2
            self.linear.bias.data[:] = 3

        def forward(self, x):
            def true_fn(x):
                return self.linear(x)

            def false_fn(x):
                return -x

            x = self.linear(x)
            h = lax.cond(x.sum().long() % 2 == 0, true_fn, false_fn, x)
            return h

    model = Net()
    x = torch.tensor([[0], [1]]).float()
    output_dir = _helper(
        model,
        x,
        'cond',
        input_names=("x",),
        enable_onnx_checker=False,
        use_pfto=False,
        do_constant_folding=False,
    )

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    assert len([x for x in actual_onnx.graph.node if x.op_type == "If"]) == 1
    ort_session = _ort_session(os.path.join(output_dir, "model.onnx"))
    actual = ort_session.run(None, {"x": x.cpu().numpy()})
    expected = model(x)
    torch.testing.assert_close(expected, torch.tensor(actual[0]))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_lax_multiple_times():
    if not pytorch_pfn_extras.requires('1.8.0'):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    if pytorch_pfn_extras.requires('1.10.0') and sys.platform == 'win32':
        pytest.skip('ONNX grad test does not work in windows CI for torch >= 1.10')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Linear(1, 1)
            self.linear.weight.data[:] = 2
            self.linear.bias.data[:] = 3

        def forward(self, x):
            def body0(it, h):
                return it * self.linear(h)

            def body1(it, h):
                return -it * self.linear(h)

            h0 = lax.fori_loop(1, 4, body0, x)
            h1 = lax.fori_loop(5, 10, body1, x)
            return h0 + h1

    model = Net()
    x = torch.tensor([[0], [1]]).float()
    output_dir = _helper(
        model,
        x,
        'lax_multiple_times',
        input_names=("x",),
        enable_onnx_checker=False,
        use_pfto=False,
        do_constant_folding=False,
    )

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    assert len([x for x in actual_onnx.graph.node if x.op_type == "Loop"]) == 2
    ort_session = _ort_session(os.path.join(output_dir, "model.onnx"))
    actual = ort_session.run(None, {"x": x.cpu().numpy()})
    expected = model(x)
    torch.testing.assert_close(expected, torch.tensor(actual[0]))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_lax_nested():
    if not pytorch_pfn_extras.requires('1.8.0'):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    if pytorch_pfn_extras.requires('1.10.0') and sys.platform == 'win32':
        pytest.skip('ONNX grad test does not work in windows CI for torch >= 1.10')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Linear(1, 1)
            self.linear.weight.data[:] = 2
            self.linear.bias.data[:] = 3

        def forward(self, x):
            def body0(it, h):
                h = it * h

                def body1(it, h):
                    return self.linear(h) + it

                return lax.fori_loop(1, 4, body1, h)

            h = lax.fori_loop(1, 3, body0, x)
            return h + 1

    model = Net()
    x = torch.tensor([[0], [1]]).float()
    output_dir = _helper(
        model,
        x,
        'lax_nested',
        input_names=("x",),
        enable_onnx_checker=False,
        use_pfto=False,
        do_constant_folding=False,
    )

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    assert len([x for x in actual_onnx.graph.node if x.op_type == "Loop"]) == 1
    loop_node = [x for x in actual_onnx.graph.node if x.op_type == "Loop"][0]
    assert len([x for x in loop_node.attribute[0].g.node if x.op_type == "Loop"]) == 1
    ort_session = _ort_session(os.path.join(output_dir, "model.onnx"))
    actual = ort_session.run(None, {"x": x.cpu().numpy()})
    expected = model(x)
    torch.testing.assert_close(expected, torch.tensor(actual[0]))
