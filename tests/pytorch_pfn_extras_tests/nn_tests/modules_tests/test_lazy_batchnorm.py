import torch
from torch import nn

from pytorch_pfn_extras.nn import LazyBatchNorm1d, LazyBatchNorm2d, LazyBatchNorm3d  # NOQA

from tests.pytorch_pfn_extras_tests.nn_tests.modules_tests.test_lazy import \
    LazyTestBase


class TestLazyBatchNorm1d(LazyTestBase):

    def get_original_module(self):
        return nn.BatchNorm1d(10)

    def get_lazy_module(self):
        return LazyBatchNorm1d(None)

    def get_input(self):
        return torch.rand(10, 10)


class TestLazyBatchNorm2d(LazyTestBase):

    def get_original_module(self):
        return nn.BatchNorm2d(10)

    def get_lazy_module(self):
        return LazyBatchNorm2d(None)

    def get_input(self):
        return torch.rand(10, 10, 10, 10)


class TestLazyBatchNorm3d(LazyTestBase):

    def get_original_module(self):
        return nn.BatchNorm3d(10)

    def get_lazy_module(self):
        return LazyBatchNorm3d(None)

    def get_input(self):
        return torch.rand(10, 10, 10, 10, 10)
