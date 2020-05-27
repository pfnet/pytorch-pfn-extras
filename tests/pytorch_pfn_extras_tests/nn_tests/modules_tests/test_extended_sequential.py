import unittest
import pytest
import functools

import numpy
import torch
from torch import nn
import pytorch_pfn_extras as ppe

assertions = unittest.TestCase('__init__')


class UserDefinedLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


@pytest.mark.parametrize('container', [
    nn.Sequential,
    nn.ModuleList,
    nn.ModuleDict,
])
@pytest.mark.parametrize('irregular_layer', [
    UserDefinedLayer,
    # No reset_parameters
    nn.ReLU,
    # use reset_running_stats
    functools.partial(
        nn.BatchNorm1d, 1),
    # use _reset_parameters
    functools.partial(
        nn.MultiheadAttention, 1, 1),
    # ppe.nn layer
    functools.partial(
        ppe.nn.LazyConv1d, None, 1, 1),
])
class TestExtendedSequential(object):

    @pytest.fixture(autouse=True)
    def setUp(self, container, irregular_layer):
        self.l1 = ppe.nn.LazyLinear(None, 3)
        self.l2 = nn.Linear(3, 2)
        self.l3 = nn.Linear(2, 3)
        # a layer without reset_parameters
        self.l4 = irregular_layer()
        # s1: l1 -> l2
        if container == nn.Sequential:
            self.s1 = container(self.l1, self.l2)
        elif container == nn.ModuleDict:
            self.s1 = container({
                'l1': self.l1,
                'l2': self.l2})
        else:
            self.s1 = container([self.l1, self.l2])
        self.container = container
        # s2: s1 (l1 -> l2) -> l3 -> l4
        self.s2 = ppe.nn.ExtendedSequential(self.s1, self.l3, self.l4)

    def test_repeat_with_init(self):
        # s2 ((l1 -> l2) -> l3 -> l4) -> s2 ((l1 -> l2) -> l3 -> l4)
        ret = self.s2.repeat(2)
        assertions.assertIsNot(ret[0], self.s2)
        assertions.assertIs(type(ret[0]), type(self.s2))
        assertions.assertIsNot(ret[1], self.s2)
        assertions.assertIs(type(ret[1]), type(self.s2))

        # bias is filled with 0, so they should have the same values
        if self.container == nn.ModuleDict:
            numpy.testing.assert_array_equal(
                ret[0][0]['l1'].bias.detach().numpy(),
                ret[1][0]['l1'].bias.detach().numpy())
        else:
            numpy.testing.assert_array_equal(
                ret[0][0][0].bias.detach().numpy(),
                ret[1][0][0].bias.detach().numpy())
        # weight is initialized randomly, so they should be different
        assertions.assertFalse(
            numpy.array_equal(ret[0][1].weight.detach().numpy(),
                              self.l3.weight.detach().numpy()))
        # And the object should also be different
        assertions.assertIsNot(ret[0][1].weight.detach().numpy(),
                               self.l3.weight.detach().numpy())
        # Repeated elements should be different objects
        assertions.assertIsNot(ret[0], ret[1])
        # Also for the arrays
        assertions.assertIsNot(ret[0][1].weight.detach().numpy(),
                               ret[1][1].weight.detach().numpy())
        # And values should be different
        assertions.assertFalse(
            numpy.array_equal(ret[0][1].weight.detach().numpy(),
                              ret[1][1].weight.detach().numpy()))

        assertions.assertEqual(len(ret), 2)
        ret = self.s2.repeat(0, mode='init')
        assertions.assertEqual(len(ret), 0)

    def test_repeat_with_copy(self):
        # s2 ((l1 -> l2) -> l3 -> l4) -> s2 ((l1 -> l2) -> l3 -> l4)
        ret = self.s2.repeat(2, mode='copy')
        assertions.assertIsNot(ret[0], self.s2)
        assertions.assertIs(type(ret[0]), type(self.s2))
        assertions.assertIsNot(ret[1], self.s2)
        assertions.assertIs(type(ret[1]), type(self.s2))
        assertions.assertIsNot(ret[0], ret[1])

        # b is filled with 0, so they should have the same values
        if self.container == nn.ModuleDict:
            numpy.testing.assert_array_equal(
                ret[0][0]["l1"].bias.detach().numpy(),
                ret[1][0]["l1"].bias.detach().numpy())
        else:
            numpy.testing.assert_array_equal(
                ret[0][0][0].bias.detach().numpy(),
                ret[1][0][0].bias.detach().numpy())
        # W is shallowy copied, so the values should be same
        numpy.testing.assert_array_equal(
            ret[0][1].weight.detach().numpy(), self.l3.weight.detach().numpy())
        # But the object should be different
        assertions.assertIsNot(ret[0][1].weight, self.l3.weight)
        # Repeated elements should be different objects
        assertions.assertIsNot(ret[0][0], ret[1][0])
        # Also for the arrays
        assertions.assertIsNot(ret[0][1].weight, ret[1][1].weight)
        # But the values should be same
        numpy.testing.assert_array_equal(
            ret[0][1].weight.detach().numpy(),
            ret[1][1].weight.detach().numpy())

        assertions.assertEqual(len(ret), 2)
        ret = self.s2.repeat(0, mode='copy')
        assertions.assertEqual(len(ret), 0)

    def test_repeat_with_share(self):
        # s2 ((l1 -> l2) -> l3 -> l4) -> s2 ((l1 -> l2) -> l3 -> l4)
        ret = self.s2.repeat(2, mode='share')
        assertions.assertIsNot(ret[0], self.s2)
        assertions.assertIs(type(ret[0]), type(self.s2))
        assertions.assertIsNot(ret[1], self.s2)
        assertions.assertIs(type(ret[1]), type(self.s2))

        # b is filled with 0, so they should have the same values
        if self.container == nn.ModuleDict:
            numpy.testing.assert_array_equal(
                ret[0][0]["l1"].bias.detach().numpy(),
                ret[1][0]["l1"].bias.detach().numpy())
        else:
            numpy.testing.assert_array_equal(
                ret[0][0][0].bias.detach().numpy(),
                ret[1][0][0].bias.detach().numpy())
        # W is shallowy copied, so the values should be same
        numpy.testing.assert_array_equal(
            ret[0][1].weight.detach().numpy(), self.l3.weight.detach().numpy())
        numpy.testing.assert_array_equal(
            ret[1][1].weight.detach().numpy(), self.l3.weight.detach().numpy())
        # And the object should also be same
        assertions.assertIs(ret[0][1].weight, self.l3.weight)
        assertions.assertIs(ret[1][1].weight, self.l3.weight)
        # Repeated element itself should be different
        assertions.assertIsNot(ret[0], ret[1])

        assertions.assertEqual(len(ret), 2)
        ret = self.s2.repeat(0, mode='share')
        assertions.assertEqual(len(ret), 0)


class UserDefinedLayerWithReset(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    def reset_parameters(self):
        pass


class UserDefinedLayerWithUnderScoreReset(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

    def _reset_parameters(self):
        pass


class UserDefinedLayerWithParameters(nn.Module):
    def __init__(self):
        super().__init__()
        param = nn.Parameter(torch.zeros(1, 1))
        self.register_parameter('weight', param)

    def forward(self):
        pass


class UserDefinedLayerWithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.zeros(1, 1))

    def forward(self):
        pass


@pytest.mark.parametrize('module', [
    # buit-in, no parameters
    nn.ReLU,
    # no parameters
    UserDefinedLayer,
    # has `_reset_parameters`
    UserDefinedLayerWithUnderScoreReset,
    # has `reset_parameters`
    UserDefinedLayerWithReset,
])
def test_no_warning_when_repeat(module):
    model = ppe.nn.ExtendedSequential(module())
    # no warnings are raised on these modules
    with pytest.warns(None):
        model.repeat(2)


@pytest.mark.parametrize('module', [
    UserDefinedLayerWithParameters,
    UserDefinedLayerWithBuffer,
])
def test_warning_when_repeat(module):
    model = ppe.nn.ExtendedSequential(module())
    # warnings are raised on these modules
    with pytest.warns(UserWarning):
        model.repeat(2)
