import os
import sys

import onnx
import onnx.checker
import onnx.numpy_helper
import pytest
import pytorch_pfn_extras
import torch
import torch.nn as nn
import torch.onnx

from pytorch_pfn_extras.onnx import grad
from tests.pytorch_pfn_extras_tests.onnx_tests.test_export_testcase import _helper


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_grad_no_export():
    if not pytorch_pfn_extras.requires("1.8.0"):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.linear = nn.Linear(32, 20, bias=False)

        def forward(self, x):
            x = x * 0.5
            x.requires_grad_(True)
            h = self.conv(x)
            grad_x = grad(
                h,
                (x,),
                retain_graph=True,
                create_graph=True,
            )[0]
            h = self.linear(grad_x)
            return h

    model = Net()
    x = torch.ones((1, 1, 32, 32))
    y = model(x)
    assert y.shape == (1, 1, 32, 20)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_grad():
    if not pytorch_pfn_extras.requires('1.8.0'):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    if not pytorch_pfn_extras.requires('1.9.0') and sys.platform == 'win32':
        pytest.skip('ONNX grad test does not work in windows CI for torch > 1.9')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.linear = nn.Linear(32, 20, bias=False)

        def forward(self, x):
            x = x * 0.5
            x.requires_grad_(True)
            h = self.conv(x)
            grad_x = grad(
                h,
                (x,),
                retain_graph=True,
                create_graph=True,
            )
            h = self.linear(grad_x[0])
            return h

    model = Net()
    x = torch.ones((1, 1, 32, 32))
    output_dir = _helper(
        model,
        x,
        'grad',
        enable_onnx_checker=False,
    )

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    named_nodes = {n.name: n for n in actual_onnx.graph.node}
    assert 'Conv_2' in named_nodes
    assert 'Gradient_3' in named_nodes
    assert 'MatMul_5' in named_nodes

    assert list([v.name for v in actual_onnx.graph.output]) == [
        "v9_MatMul", "Gradient_y_0", "Gradient_x_0_0"
    ]
    assert named_nodes["Conv_2"].input[0] == "Gradient_x_0_0"
    assert named_nodes["Conv_2"].output[0] == "Gradient_y_0"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_grad_multiple_times():
    if not pytorch_pfn_extras.requires("1.8.0"):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    if not pytorch_pfn_extras.requires('1.9.0') and sys.platform == 'win32':
        pytest.skip('ONNX grad test does not work in windows CI for torch > 1.9')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.linear = nn.Linear(32, 20, bias=False)

        def forward(self, x):
            x0 = x * 0.5
            x0.requires_grad_(True)
            h = self.conv(x0)
            grad_x0 = grad(
                h,
                (x0,),
                retain_graph=True,
                create_graph=True,
            )
            x1 = x * 0.5
            x1.requires_grad_(True)
            h = self.conv(x1)
            grad_x1 = grad(
                h,
                (x1,),
                retain_graph=True,
                create_graph=True,
            )
            h = self.linear(grad_x0[0] + grad_x1[0])
            return h

    model = Net()
    x = torch.ones((1, 1, 32, 32))
    output_dir = _helper(
        model,
        x,
        'grad',
        enable_onnx_checker=False,
    )

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    named_nodes = {n.name: n for n in actual_onnx.graph.node}
    assert 'Conv_2' in named_nodes
    assert 'Conv_6' in named_nodes
    assert 'Gradient_3' in named_nodes
    assert 'Gradient_7' in named_nodes
    assert 'MatMul_10' in named_nodes

    assert list([v.name for v in actual_onnx.graph.output]) == [
        "v14_MatMul", "Gradient_y_0", "Gradient_x_0_0", "Gradient_y_1", "Gradient_x_0_1"
    ]
    assert named_nodes["Conv_2"].input[0] == "Gradient_x_0_0"
    assert named_nodes["Conv_2"].output[0] == "Gradient_y_0"
    assert named_nodes["Conv_6"].input[0] == "Gradient_x_0_1"
    assert named_nodes["Conv_6"].output[0] == "Gradient_y_1"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_grad_with_multiple_inputs():
    if not pytorch_pfn_extras.requires("1.8.0"):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    if not pytorch_pfn_extras.requires('1.9.0') and sys.platform == 'win32':
        pytest.skip('ONNX grad test does not work in windows CI for torch > 1.9')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(2, 6, 3)
            self.linear = nn.Linear(32, 20, bias=False)

        def forward(self, x):
            x0 = x * 0.5
            x1 = x * 2.0
            x0.requires_grad_(True)
            x1.requires_grad_(True)
            h = self.conv(torch.cat([x0, x1], dim=1))
            grad_x = grad(
                h,
                (x0, x1),
                retain_graph=True,
                create_graph=True,
            )
            h = self.linear(grad_x[0])
            return h

    model = Net()
    x = torch.ones((1, 1, 32, 32))
    output_dir = _helper(
        model,
        x,
        'grad',
        enable_onnx_checker=False,
    )

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    named_nodes = {n.name: n for n in actual_onnx.graph.node}
    assert 'Conv_5' in named_nodes
    assert 'Gradient_6' in named_nodes
    assert 'MatMul_8' in named_nodes

    assert list([v.name for v in actual_onnx.graph.output]) == [
        "v13_MatMul", "Gradient_y_0", "Gradient_x_0_0", "Gradient_x_1_0"
    ]
    assert named_nodes["Concat_4"].input[0] == "Gradient_x_0_0"
    assert named_nodes["Concat_4"].input[1] == "Gradient_x_1_0"
    assert named_nodes["Conv_5"].output[0] == "Gradient_y_0"
