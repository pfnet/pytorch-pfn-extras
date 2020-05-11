import unittest
import numpy
from torch import nn
import pytorch_pfn_extras as ppe


class TestExtendedSequential(unittest.TestCase):
    def setUp(self):
        self.l1 = ppe.nn.LazyLinear(None, 3)
        self.l2 = nn.Linear(3, 2)
        self.l3 = nn.Linear(2, 3)
        # s1: l1 -> l2
        self.s1 = ppe.nn.ExtendedSequential(self.l1, self.l2)
        # s2: s1 (l1 -> l2) -> l3
        self.s2 = ppe.nn.ExtendedSequential(self.s1, self.l3)

    def test_repeat_with_init(self):
        # s2 ((l1 -> l2) -> l3) -> s2 ((l1 -> l2) -> l3)
        ret = self.s2.repeat(2)
        self.assertIsNot(ret[0], self.s2)
        self.assertIs(type(ret[0]), type(self.s2))
        self.assertIsNot(ret[1], self.s2)
        self.assertIs(type(ret[1]), type(self.s2))

        # bias is filled with 0, so they should have the same values
        numpy.testing.assert_array_equal(
            ret[0][0][0].bias.detach().numpy(),
            ret[1][0][0].bias.detach().numpy())
        # weight is initialized randomly, so they should be different
        self.assertFalse(
            numpy.array_equal(ret[0][1].weight.detach().numpy(),
                              self.l3.weight.detach().numpy()))
        # And the object should also be different
        self.assertIsNot(ret[0][1].weight.detach().numpy(),
                         self.l3.weight.detach().numpy())
        # Repeated elements should be different objects
        self.assertIsNot(ret[0], ret[1])
        # Also for the arrays
        self.assertIsNot(ret[0][1].weight.detach().numpy(),
                         ret[1][1].weight.detach().numpy())
        # And values should be different
        self.assertFalse(
            numpy.array_equal(ret[0][1].weight.detach().numpy(),
                              ret[1][1].weight.detach().numpy()))

        self.assertEqual(len(ret), 2)
        ret = self.s2.repeat(0, mode='init')
        self.assertEqual(len(ret), 0)

    def test_repeat_with_copy(self):
        # s2 ((l1 -> l2) -> l3) -> s2 ((l1 -> l2) -> l3)
        ret = self.s2.repeat(2, mode='copy')
        self.assertIsNot(ret[0], self.s2)
        self.assertIs(type(ret[0]), type(self.s2))
        self.assertIsNot(ret[1], self.s2)
        self.assertIs(type(ret[1]), type(self.s2))
        self.assertIsNot(ret[0], ret[1])

        # b is filled with 0, so they should have the same values
        numpy.testing.assert_array_equal(
            ret[0][0][0].bias.detach().numpy(),
            ret[1][0][0].bias.detach().numpy())
        # W is shallowy copied, so the values should be same
        numpy.testing.assert_array_equal(
            ret[0][1].weight.detach().numpy(), self.l3.weight.detach().numpy())
        # But the object should be different
        self.assertIsNot(ret[0][1].weight, self.l3.weight)
        # Repeated elements should be different objects
        self.assertIsNot(ret[0][0], ret[1][0])
        # Also for the arrays
        self.assertIsNot(ret[0][1].weight, ret[1][1].weight)
        # But the values should be same
        numpy.testing.assert_array_equal(
            ret[0][1].weight.detach().numpy(),
            ret[1][1].weight.detach().numpy())

        self.assertEqual(len(ret), 2)
        ret = self.s2.repeat(0, mode='copy')
        self.assertEqual(len(ret), 0)

    def test_repeat_with_share(self):
        # s2 ((l1 -> l2) -> l3) -> s2 ((l1 -> l2) -> l3)
        ret = self.s2.repeat(2, mode='share')
        self.assertIsNot(ret[0], self.s2)
        self.assertIs(type(ret[0]), type(self.s2))
        self.assertIsNot(ret[1], self.s2)
        self.assertIs(type(ret[1]), type(self.s2))

        # b is filled with 0, so they should have the same values
        numpy.testing.assert_array_equal(
            ret[0][0][0].bias.detach().numpy(),
            ret[1][0][0].bias.detach().numpy())
        # W is shallowy copied, so the values should be same
        numpy.testing.assert_array_equal(
            ret[0][1].weight.detach().numpy(), self.l3.weight.detach().numpy())
        numpy.testing.assert_array_equal(
            ret[1][1].weight.detach().numpy(), self.l3.weight.detach().numpy())
        # And the object should also be same
        self.assertIs(ret[0][1].weight, self.l3.weight)
        self.assertIs(ret[1][1].weight, self.l3.weight)
        # Repeated element itself should be different
        self.assertIsNot(ret[0], ret[1])

        self.assertEqual(len(ret), 2)
        ret = self.s2.repeat(0, mode='share')
        self.assertEqual(len(ret), 0)
