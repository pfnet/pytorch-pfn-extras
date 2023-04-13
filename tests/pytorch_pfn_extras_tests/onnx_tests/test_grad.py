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
from pytorch_pfn_extras_tests.onnx_tests.test_export_testcase import _helper


def _get_name(onnx_graph: onnx.GraphProto, output_name: str):
    in_name = None
    out_name = None

    for node in onnx_graph.node:
        if node.output[0] == output_name:
            assert node.op_type == "Identity"
            in_name = node.input[0]
        if len(node.input) == 0:
            continue
        if node.input[0] == output_name:
            assert node.op_type == "Identity"
            out_name = node.output[0]

    return in_name, out_name


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


# @pytest.mark.parametrize("use_pfto", [False, True])
@pytest.mark.filterwarnings("ignore:The shape inference of ai.onnx.preview..Gradient type is missing:UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_grad(use_pfto: bool = False):
    if not pytorch_pfn_extras.requires('1.8.0'):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    if pytorch_pfn_extras.requires('1.10.0') and sys.platform == 'win32':
        pytest.skip('ONNX grad test does not work in windows CI for torch >= 1.10')

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
        use_pfto=use_pfto,
    )

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    named_nodes = {n.name: n for n in actual_onnx.graph.node}
    if pytorch_pfn_extras.requires("1.13"):
        assert '/_ppe_as_out_module/conv/Conv' in named_nodes
        assert '/_ppe_as_out_module/Gradient' in named_nodes
        assert '/_ppe_as_out_module/linear/MatMul' in named_nodes
    else:
        assert 'Conv_2' in named_nodes
        assert 'Gradient_4' in named_nodes
        assert 'MatMul_6' in named_nodes

    if use_pfto:
        assert list([v.name for v in actual_onnx.graph.output]) == [
            "linear.72", "Gradient_y_0", "Gradient_x_0_0"
        ]
        y_in, _ = _get_name(actual_onnx.graph, "input.1")
        if pytorch_pfn_extras.requires("1.13"):
            assert named_nodes["/_ppe_as_out_module/conv/Conv"].input[0] == "Gradient_x_0_0"
            assert named_nodes["/_ppe_as_out_module/conv/Conv"].output[0] == y_in
        else:
            assert named_nodes["Conv_2"].input[0] == "Gradient_x_0_0"
            assert named_nodes["Conv_2"].output[0] == y_in
    else:
        assert list([v.name for v in actual_onnx.graph.output]) == [
            "v10_MatMul", "Gradient_y_0", "Gradient_x_0_0"
        ]
        y_in, _ = _get_name(actual_onnx.graph, "Gradient_y_0")
        if pytorch_pfn_extras.requires("1.13"):
            assert named_nodes["/_ppe_as_out_module/conv/Conv"].input[0] == "Gradient_x_0_0"
            assert named_nodes["/_ppe_as_out_module/conv/Conv"].output[0] == y_in
        else:
            assert named_nodes["Conv_2"].input[0] == "Gradient_x_0_0"
            assert named_nodes["Conv_2"].output[0] == y_in


@pytest.mark.parametrize("use_pfto", [False, True])
@pytest.mark.filterwarnings("ignore:The shape inference of ai.onnx.preview..Gradient type is missing:UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_grad_multiple_times(use_pfto: bool):
    if not pytorch_pfn_extras.requires("1.8.0"):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    if pytorch_pfn_extras.requires('1.10.0') and sys.platform == 'win32':
        pytest.skip('ONNX grad test does not work in windows CI for torch >= 1.10')

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
        use_pfto=False,
    )

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    named_nodes = {n.name: n for n in actual_onnx.graph.node}
    if pytorch_pfn_extras.requires("1.13"):
        assert '/_ppe_as_out_module/conv/Conv' in named_nodes
        assert '/_ppe_as_out_module/conv_1/Conv' in named_nodes
        assert '/_ppe_as_out_module/Gradient' in named_nodes
        assert '/_ppe_as_out_module/Gradient_1' in named_nodes
        assert '/_ppe_as_out_module/linear/MatMul' in named_nodes
    else:
        assert 'Conv_2' in named_nodes
        assert 'Conv_7' in named_nodes
        assert 'Gradient_4' in named_nodes
        assert 'Gradient_9' in named_nodes
        assert 'MatMul_12' in named_nodes

    if use_pfto:
        assert list([v.name for v in actual_onnx.graph.output]) == [
            "v16_MatMul", "Gradient_y_0", "Gradient_x_0_0", "Gradient_y_1", "Gradient_x_0_1"
        ]
        y0_in, _ = _get_name(actual_onnx.graph, "Gradient_y_0")
        y1_in, _ = _get_name(actual_onnx.graph, "Gradient_y_1")
        if pytorch_pfn_extras.requires("1.13"):
            assert named_nodes["/_ppe_as_out_module/conv/Conv"].input[0] == "Gradient_x_0_0"
            assert named_nodes["/_ppe_as_out_module/conv/Conv"].output[0] == y0_in
            assert named_nodes["/_ppe_as_out_module/conv_1/Conv"].input[0] == "Gradient_x_0_1"
            assert named_nodes["/_ppe_as_out_module/conv_1/Conv"].output[0] == y1_in
        else:
            assert named_nodes["Conv_2"].input[0] == "Gradient_x_0_0"
            assert named_nodes["Conv_2"].output[0] == y0_in
            assert named_nodes["Conv_7"].input[0] == "Gradient_x_0_1"
            assert named_nodes["Conv_7"].output[0] == y1_in
    else:
        assert list([v.name for v in actual_onnx.graph.output]) == [
            "v16_MatMul", "Gradient_y_0", "Gradient_x_0_0", "Gradient_y_1", "Gradient_x_0_1"
        ]
        y0_in, _ = _get_name(actual_onnx.graph, "Gradient_y_0")
        y1_in, _ = _get_name(actual_onnx.graph, "Gradient_y_1")
        if pytorch_pfn_extras.requires("1.13"):
            assert named_nodes["/_ppe_as_out_module/conv/Conv"].input[0] == "Gradient_x_0_0"
            assert named_nodes["/_ppe_as_out_module/conv/Conv"].output[0] == y0_in
            assert named_nodes["/_ppe_as_out_module/conv_1/Conv"].input[0] == "Gradient_x_0_1"
            assert named_nodes["/_ppe_as_out_module/conv_1/Conv"].output[0] == y1_in
        else:
            assert named_nodes["Conv_2"].input[0] == "Gradient_x_0_0"
            assert named_nodes["Conv_2"].output[0] == y0_in
            assert named_nodes["Conv_7"].input[0] == "Gradient_x_0_1"
            assert named_nodes["Conv_7"].output[0] == y1_in


# @pytest.mark.parametrize("use_pfto", [False, True])
@pytest.mark.filterwarnings("ignore:The shape inference of ai.onnx.preview..Gradient type is missing:UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_grad_with_multiple_inputs(use_pfto: bool = False):
    if not pytorch_pfn_extras.requires("1.8.0"):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    if pytorch_pfn_extras.requires('1.10.0') and sys.platform == 'win32':
        pytest.skip('ONNX grad test does not work in windows CI for torch >= 1.10')

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
        use_pfto=use_pfto,
    )

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    named_nodes = {n.name: n for n in actual_onnx.graph.node}
    if pytorch_pfn_extras.requires("1.13"):
        assert '/_ppe_as_out_module/conv/Conv' in named_nodes
        assert '/_ppe_as_out_module/Gradient' in named_nodes
        assert '/_ppe_as_out_module/linear/MatMul' in named_nodes
    else:
        assert 'Conv_5' in named_nodes
        assert 'Gradient_7' in named_nodes
        assert 'MatMul_9' in named_nodes

    if use_pfto:
        assert list([v.name for v in actual_onnx.graph.output]) == [
            "linear.87", "Gradient_y_0", "Gradient_x_0_0", "Gradient_x_1_0"
        ]
        y_in, _ = _get_name(actual_onnx.graph, "Gradient_y_0")
        if pytorch_pfn_extras.requires("1.13"):
            assert named_nodes["/_ppe_as_out_module/Concat"].input[0] == "Gradient_x_0_0"
            assert named_nodes["/_ppe_as_out_module/Concat"].input[1] == "Gradient_x_1_0"
            assert named_nodes["/_ppe_as_out_module/conv/Conv"].output[0] == y_in
        else:
            assert named_nodes["Concat_4"].input[0] == "x0"
            assert named_nodes["Concat_4"].input[1] == "x1"
            assert named_nodes["Conv_5"].output[0] == "conv.output.1"
    else:
        assert list([v.name for v in actual_onnx.graph.output]) == [
            "v14_MatMul", "Gradient_y_0", "Gradient_x_0_0", "Gradient_x_1_0"
        ]
        y_in, _ = _get_name(actual_onnx.graph, "Gradient_y_0")
        if pytorch_pfn_extras.requires("1.13"):
            assert named_nodes["/_ppe_as_out_module/Concat"].input[0] == "Gradient_x_0_0"
            assert named_nodes["/_ppe_as_out_module/Concat"].input[1] == "Gradient_x_1_0"
            assert named_nodes["/_ppe_as_out_module/conv/Conv"].output[0] == y_in
        else:
            assert named_nodes["Concat_4"].input[0] == "Gradient_x_0_0"
            assert named_nodes["Concat_4"].input[1] == "Gradient_x_1_0"
            assert named_nodes["Conv_5"].output[0] == y_in
