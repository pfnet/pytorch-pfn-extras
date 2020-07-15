from collections import OrderedDict
from contextlib import suppress
import os

import numpy as np
import onnx
import onnx.checker
import onnx.numpy_helper
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

from pytorch_pfn_extras.onnx import annotate
from pytorch_pfn_extras.onnx import apply_annotation
from pytorch_pfn_extras.onnx import export_testcase
from pytorch_pfn_extras.onnx import scoped_anchor
from tests.pytorch_pfn_extras_tests.onnx.test_export_testcase import _helper


def test_annotate():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 12, 3)
            self.linear = nn.Linear(28, 10)
            self.linear2 = nn.Linear(10, 5)

        def forward(self, x):
            with annotate(aaa='a', bbb=['b', 'c']):
                h = self.conv(x)
            h = self.conv2(h)
            with annotate(zzz=99, yyy=[9, 9]):
                h = self.linear(h)
                h = self.linear2(h)
            return h

    model = Net()
    x = torch.ones((1, 1, 32, 32))
    output_dir = _helper(model, x, 'annotate')

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    named_nodes = {n.name: n for n in actual_onnx.graph.node}
    assert 'Conv_0' in named_nodes
    assert 'Conv_1' in named_nodes
    assert 'MatMul_2' in named_nodes
    assert 'MatMul_4' in named_nodes

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
    node_matmul_4_attrs = [a.name for a in named_nodes['MatMul_4'].attribute]
    assert 'aaa' not in node_matmul_4_attrs
    assert 'bbb' not in node_matmul_4_attrs
    assert 'zzz' in node_matmul_4_attrs
    assert 'yyy' in node_matmul_4_attrs


def test_apply_annotation():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 12, 3)
            self.linear = nn.Linear(28, 10)
            self.linear2 = nn.Linear(10, 5)

        def forward(self, x):
            def _fn1():
                h = self.conv(x)
                h = F.relu(h)
                return h
            h = apply_annotation(_fn1, aaa='a', bbb=['b', 'c'])
            h = self.conv2(h)

            def _fn2(x):
                h = self.linear(x)
                h = self.linear2(h)
                h = F.elu(h)
                return h
            h = apply_annotation(_fn2, h, zzz=99, yyy=[9, 9])
            return h

    model = Net()
    x = torch.ones((1, 1, 32, 32))
    output_dir = _helper(model, x, 'apply_annotation')

    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    named_nodes = {n.name: n for n in actual_onnx.graph.node}
    assert 'Conv_0' in named_nodes
    assert 'Relu_1' in named_nodes
    assert 'Conv_2' in named_nodes
    assert 'MatMul_3' in named_nodes
    assert 'MatMul_5' in named_nodes
    assert 'Elu_7' in named_nodes

    node_attrs = [a.name for a in named_nodes['Conv_0'].attribute]
    assert 'aaa' in node_attrs
    assert 'bbb' in node_attrs
    assert 'zzz' not in node_attrs
    assert 'yyy' not in node_attrs
    node_attrs = [a.name for a in named_nodes['Relu_1'].attribute]
    assert 'aaa' in node_attrs
    assert 'bbb' in node_attrs
    assert 'zzz' not in node_attrs
    assert 'yyy' not in node_attrs
    node_attrs = [a.name for a in named_nodes['Conv_2'].attribute]
    assert 'aaa' not in node_attrs
    assert 'bbb' not in node_attrs
    assert 'zzz' not in node_attrs
    assert 'yyy' not in node_attrs
    node_attrs = [a.name for a in named_nodes['MatMul_3'].attribute]
    assert 'aaa' not in node_attrs
    assert 'bbb' not in node_attrs
    assert 'zzz' in node_attrs
    assert 'yyy' in node_attrs
    node_attrs = [a.name for a in named_nodes['MatMul_5'].attribute]
    assert 'aaa' not in node_attrs
    assert 'bbb' not in node_attrs
    assert 'zzz' in node_attrs
    assert 'yyy' in node_attrs
    node_attrs = [a.name for a in named_nodes['Elu_7'].attribute]
    assert 'aaa' not in node_attrs
    assert 'bbb' not in node_attrs
    assert 'zzz' in node_attrs
    assert 'yyy' in node_attrs


def test_scoped_anchor():
    class Net(nn.Module):
        def __init__(self, anchor_mode='on'):
            super(Net, self).__init__()

            self.conv = nn.Conv2d(6, 9, 3)
            self.conv2 = nn.Conv2d(9, 12, 3)
            self.linear = nn.Linear(28, 20)
            self.linear2 = nn.Linear(20, 15)
            self.gn = nn.GroupNorm(3, 12)  # to check multiple nodes
            self.linear3 = nn.Linear(15, 10)

            # to check output values (not reduce node number)
            nn.init.constant_(self.conv.weight, 0.1)
            nn.init.constant_(self.conv.bias, 0.1)
            nn.init.constant_(self.conv2.weight, 0.1)
            nn.init.constant_(self.conv2.bias, 0.1)
            nn.init.constant_(self.linear.weight, 0.1)
            nn.init.constant_(self.linear.bias, 0.1)
            nn.init.constant_(self.linear2.weight, 0.1)
            nn.init.constant_(self.linear2.bias, 0.1)
            nn.init.constant_(self.linear3.weight, 0.1)
            nn.init.constant_(self.linear3.bias, 0.1)

            if anchor_mode == 'on':
                self.anchor1 = scoped_anchor(aaa='a', bbb=['b', 'c'])
                self.anchor2 = scoped_anchor(ccc=[1, 2])
            elif anchor_mode == 'no_param':
                self.anchor1 = scoped_anchor()
                self.anchor2 = scoped_anchor()
            else:
                self.anchor1 = suppress()
                self.anchor2 = suppress()

        def forward(self, x):
            h = self.conv(x)
            with self.anchor1:
                h = self.conv2(h)
                h = self.linear(h)
            h = self.linear2(h)
            with self.anchor2:
                h = self.gn(h)
            h = self.linear3(h)
            return h

    # first output graph is valid or not check
    no_param_model = Net(anchor_mode='no_param')
    x = torch.randn((1, 6, 32, 32))
    dirname = 'scoped_anchor'
    no_attr_dirname = os.path.join(dirname, 'no_attr_graph')
    no_attr_output_dir = _helper(
        no_param_model, x, no_attr_dirname, opset_version=11)
    no_attr_onnx = onnx.load(os.path.join(no_attr_output_dir, 'model.onnx'))
    try:
        onnx.checker.check_model(no_attr_onnx)
    except onnx.checker.ValidationError as e:
        pytest.fail(e)

    # make full annotated graph
    model = Net()
    output_dir = _helper(model, x, dirname, opset_version=11)

    # mak plain graph to compair with anchored graph
    no_anchor_model = Net(anchor_mode='off')
    no_anchor_dirname = os.path.join(dirname, 'no_anchor_graph')
    no_anchor_model_dir = _helper(
        no_anchor_model, x, no_anchor_dirname, opset_version=11)

    # anchored model outputs same output value with base model
    def load_tensor(path):
        with open(path, 'rb') as fp:
            tensor = onnx.TensorProto()
            tensor.ParseFromString(fp.read())
        return tensor
    actual_out = onnx.numpy_helper.to_array(load_tensor(
        os.path.join(output_dir, 'test_data_set_0', 'output_0.pb')))
    expected_out = onnx.numpy_helper.to_array(load_tensor(
        os.path.join(no_anchor_model_dir, 'test_data_set_0', 'output_0.pb')))
    np.testing.assert_allclose(expected_out, actual_out)

    # output graph check
    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    # consider python<3.6.5
    # node is expected computational order by ONNX spec
    named_nodes = OrderedDict()
    previous_node = None
    for i, node in enumerate(actual_onnx.graph.node):
        if previous_node is not None:
            named_nodes[previous_node.name] += (node,)
        named_nodes[node.name] = node, previous_node
        previous_node = node
    named_nodes[node.name] += (None,)

    assert 'Anchor_0_start' in named_nodes
    assert 'Anchor_0_end' in named_nodes
    assert 'Anchor_1_start' in named_nodes
    assert 'Anchor_1_end' in named_nodes

    anchor_node, pre_node, next_node = named_nodes['Anchor_0_start']
    anchor_attrs = [a.name for a in anchor_node.attribute]
    assert 'aaa' in anchor_attrs
    assert 'bbb' in anchor_attrs
    assert 'ccc' not in anchor_attrs
    assert pre_node.name == 'Conv_0'
    assert next_node.name == 'Conv_3'
    anchor_node, pre_node, next_node = named_nodes['Anchor_0_end']
    anchor_attrs = [a.name for a in anchor_node.attribute]
    assert 'aaa' in anchor_attrs
    assert 'bbb' in anchor_attrs
    assert 'ccc' not in anchor_attrs
    assert pre_node.name == 'Add_7'
    assert next_node.name == 'MatMul_10'
    anchor_node, pre_node, next_node = named_nodes['Anchor_1_start']
    anchor_attrs = [a.name for a in anchor_node.attribute]
    assert 'aaa' not in anchor_attrs
    assert 'bbb' not in anchor_attrs
    assert 'ccc' in anchor_attrs
    assert pre_node.name == 'Add_11'
    assert next_node.name == 'Constant_14'  # this is shape of next reshape
    anchor_node, pre_node, next_node = named_nodes['Anchor_1_end']
    anchor_attrs = [a.name for a in anchor_node.attribute]
    assert 'aaa' not in anchor_attrs
    assert 'bbb' not in anchor_attrs
    assert 'ccc' in anchor_attrs
    assert pre_node.name == 'Add_22'
    assert next_node.name == 'MatMul_25'


def test_scoped_anchor_multiple_inout():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.id = nn.Identity()

        def forward(self, *xs):
            with scoped_anchor():
                xs = self.id(xs)
                h = torch.cat(xs, 1)
                h = h.t()
                hs = h.split(1)
                hs = self.id(hs)  # to check internal dummy anchor
                hs += (hs[0], hs[1])
                h = torch.cat(hs, 0)
                hs = h.split(1)
                return self.id(hs)

    model = Net()
    x = torch.randn((4, 1))
    x = (x, x, x)
    output_dir = _helper(model, x, 'scoped_anchor_multiple_inout')
    actual_onnx = onnx.load(os.path.join(output_dir, 'model.onnx'))
    try:
        onnx.checker.check_model(actual_onnx)
    except onnx.checker.ValidationError as e:
        pytest.fail(e)

    # consider python<3.6.5
    # node is expected computational order by ONNX spec
    named_nodes = OrderedDict()
    previous_node = None
    for i, node in enumerate(actual_onnx.graph.node):
        if previous_node is not None:
            named_nodes[previous_node.name] += (node,)
        named_nodes[node.name] = node, previous_node
        previous_node = node
    named_nodes[node.name] += (None,)

    assert 'Anchor_0_start' in named_nodes
    assert 'Anchor_0_end' in named_nodes

    anchor_node, pre_node, next_node = named_nodes['Anchor_0_start']
    anchor_attrs = [a.name for a in anchor_node.attribute]
    assert pre_node is None
    assert next_node.name == 'Concat_4'
    anchor_node, pre_node, next_node = named_nodes['Anchor_0_end']
    anchor_attrs = [a.name for a in anchor_node.attribute]
    assert pre_node.name == 'Split_10'
    assert next_node is None
