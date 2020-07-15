import os

import onnx
import torch
import torch.nn as nn
import torch.onnx

from pytorch_pfn_extras.onnx import annotate
from tests.pytorch_pfn_extras_tests.onnx.test_export_testcase import _helper


def test_annotate():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 12, 3)
            self.linear = nn.Linear(28, 10)

        def forward(self, x):
            with annotate(self.conv, aaa='a', bbb=['b', 'c']):
                h = self.conv(x)
            h = self.conv2(h)
            with annotate(self.linear, zzz=99, yyy=[9, 9]):
                h = self.linear(h)
            return h

    model = Net()
    x = torch.ones((1, 1, 32, 32))
    output_dir = _helper(
        model, x, 'annotate', enable_onnx_checker=False)

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    named_nodes = {n.name: n for n in actual_onnx.graph.node}
    assert 'Conv_0' in named_nodes
    assert 'Conv_1' in named_nodes
    assert 'MatMul_2' in named_nodes

    node_conv_0_attrs = [a.name for a in named_nodes['Conv_0'].attribute]
    assert 'aaa' in node_conv_0_attrs
    assert 'bbb' in node_conv_0_attrs
    assert 'zzz' not in node_conv_0_attrs
    assert 'yyy' not in node_conv_0_attrs
    node_conv_1_attrs = [a.name for a in named_nodes['Conv_1'].attribute]
    assert 'aaa' not in node_conv_1_attrs
    assert 'bbb' not in node_conv_1_attrs
    assert 'zzz' not in node_conv_1_attrs
    assert 'yyy' not in node_conv_1_attrs
    node_matmul_2_attrs = [a.name for a in named_nodes['MatMul_2'].attribute]
    assert 'aaa' not in node_matmul_2_attrs
    assert 'bbb' not in node_matmul_2_attrs
    assert 'zzz' in node_matmul_2_attrs
    assert 'yyy' in node_matmul_2_attrs
