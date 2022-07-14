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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_as_output_no_export():
    if not pytorch_pfn_extras.requires("1.8.0"):
        pytest.skip('skip for PyTorch 1.7 or earlier')

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
    if not pytorch_pfn_extras.requires("1.8.0"):
        pytest.skip('skip for PyTorch 1.7 or earlier')

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
    assert 'Conv_0' in named_nodes
    assert 'MatMul_2' in named_nodes

    outputs = list([v.name for v in actual_onnx.graph.output])
    assert len(outputs) == 2
    assert outputs[1] == "h"
    assert named_nodes["Conv_0"].output[0] == "h"
    assert named_nodes["MatMul_2"].input[0] == "h"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_no_as_output():
    if not pytorch_pfn_extras.requires("1.8.0"):
        pytest.skip('skip for PyTorch 1.7 or earlier')

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
    assert 'Conv_0' in named_nodes
    assert 'MatMul_2' in named_nodes

    assert len([v.name for v in actual_onnx.graph.output]) == 1


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
        output_dir = _helper(model, (x, b), 'as_output', use_pfto=False)

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    named_nodes = {n.name: n for n in actual_onnx.graph.node}
    assert 'Conv_0' in named_nodes
    assert 'If_2' in named_nodes
    assert 'MatMul_6' in named_nodes

    assert len([v.name for v in actual_onnx.graph.output]) == 1
