import torch
from torch import nn

from pytorch_pfn_extras.nn import LazyConv1d, LazyConv2d, LazyConv3d

from tests.pytorch_pfn_extras_tests.nn_tests.modules_tests.test_lazy import \
    LazyTestBase


class TestLazyConv1d(LazyTestBase):

    def get_original_module(self):
        return nn.Conv1d(3, 4, 2)

    def get_lazy_module(self):
        return LazyConv1d(None, 4, 2)

    def get_input(self):
        return torch.rand(4, 3, 10)


class TestLazyConv2d(LazyTestBase):

    def get_original_module(self):
        return nn.Conv2d(3, 4, 2)

    def get_lazy_module(self):
        return LazyConv2d(None, 4, 2)

    def get_input(self):
        return torch.rand(4, 3, 10, 10)


class TestLazyConv3d(LazyTestBase):

    def get_original_module(self):
        return nn.Conv3d(3, 4, 2)

    def get_lazy_module(self):
        return LazyConv3d(None, 4, 2)

    def get_input(self):
        return torch.rand(4, 3, 10, 10, 10)
