import os

import onnx
import onnx.checker
import onnx.numpy_helper
import pytest
import pytorch_pfn_extras
import torch
import torch.nn as nn
import torch.onnx

from pytorch_pfn_extras.onnx import as_output
from pytorch_pfn_extras_tests.onnx_tests.test_export_testcase import _helper


def _get_name(onnx_graph: onnx.GraphProto, output_name: str):
    in_name = None
    out_name = None

    for node in onnx_graph.node:
        if node.output[0] == output_name:
            assert node.op_type == "Identity"
            in_name = node.input[0]
        if node.input[0] == output_name:
            assert node.op_type == "Identity"
            out_name = node.output[0]

    assert in_name is not None
    assert out_name is not None
    return in_name, out_name


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_as_output_no_export():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.linear = nn.Linear(30, 20, bias=False)

        def forward(self, x):
            h = self.conv(x)
            h = as_output("h", h)
            h = self.linear(h)
            return h

    model = Net()
    x = torch.ones((1, 1, 32, 32))
    y = model(x)
    assert y.shape == (1, 6, 30, 20)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_as_output():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.linear = nn.Linear(30, 20, bias=False)

        def forward(self, x):
            h = self.conv(x)
            h = as_output("h", h)
            h = self.linear(h)
            return h

    model = Net()
    x = torch.ones((1, 1, 32, 32))
    output_dir = _helper(model, x, 'as_output')

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    named_nodes = {n.name: n for n in actual_onnx.graph.node}
    assert '/_ppe_as_out_module/conv/Conv' in named_nodes
    assert '/_ppe_as_out_module/linear/MatMul' in named_nodes

    outputs = list([v.name for v in actual_onnx.graph.output])
    assert len(outputs) == 2
    assert outputs[1] == "h"
    in_name, out_name = _get_name(actual_onnx.graph, "h")
    assert named_nodes["/_ppe_as_out_module/conv/Conv"].output[0] == in_name
    assert named_nodes["/_ppe_as_out_module/linear/MatMul"].input[0] == out_name


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_as_output_to_input():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.linear = nn.Linear(30, 20, bias=False)

        def forward(self, x):
            x = as_output("x", x)
            h = self.conv(x)
            h = self.linear(h)
            return h

    model = Net()
    x = torch.ones((1, 1, 32, 32))
    output_dir = _helper(model, x, 'as_output')

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    named_nodes = {n.name: n for n in actual_onnx.graph.node}
    assert '/_ppe_as_out_module/conv/Conv' in named_nodes

    outputs = list([v.name for v in actual_onnx.graph.output])
    assert len(outputs) == 2
    assert outputs[1] == "x"
    _, out_name = _get_name(actual_onnx.graph, "x")
    assert named_nodes["/_ppe_as_out_module/conv/Conv"].input[0] == out_name


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_as_output_to_output():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.linear = nn.Linear(30, 20, bias=False)

        def forward(self, x):
            h = self.conv(x)
            h = self.linear(h)
            h = as_output("out", h)
            return h

    model = Net()
    x = torch.ones((1, 1, 32, 32))
    output_dir = _helper(model, x, 'as_output')

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    named_nodes = {n.name: n for n in actual_onnx.graph.node}
    assert '/_ppe_as_out_module/linear/MatMul' in named_nodes

    outputs = list([v.name for v in actual_onnx.graph.output])
    assert len(outputs) == 2
    assert outputs[1] == "out"
    in_name, _ = _get_name(actual_onnx.graph, "out")
    assert named_nodes["/_ppe_as_out_module/linear/MatMul"].output[0] == in_name


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_no_as_output():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.linear = nn.Linear(30, 20, bias=False)

        def forward(self, x):
            h = self.conv(x)
            h = self.linear(h)
            return h

    model = Net()
    x = torch.ones((1, 1, 32, 32))
    output_dir = _helper(model, x, 'as_output')

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    named_nodes = {n.name: n for n in actual_onnx.graph.node}
    assert '/_ppe_as_out_module/conv/Conv' in named_nodes
    assert '/_ppe_as_out_module/linear/MatMul' in named_nodes

    assert len([v.name for v in actual_onnx.graph.output]) == 1


@pytest.mark.xfail
def test_as_output_in_scripting():

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.linear = nn.Linear(30, 20, bias=False)

        def forward(self, x, b):
            h = self.conv(x)
            if b:  # IF statement to check scripted (not traced) IR
                h = -h
            h = as_output("h", h)
            h = self.linear(h)
            return h

    model = torch.jit.script(Net())
    x = torch.ones((1, 1, 32, 32))
    b = torch.tensor(True)
    with pytest.warns(UserWarning):
        output_dir = _helper(model, (x, b), 'as_output')

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    named_nodes = {n.name: n for n in actual_onnx.graph.node}
    assert 'Conv_0' in named_nodes
    assert 'If_2' in named_nodes
    assert 'MatMul_6' in named_nodes

    assert len([v.name for v in actual_onnx.graph.output]) == 1
