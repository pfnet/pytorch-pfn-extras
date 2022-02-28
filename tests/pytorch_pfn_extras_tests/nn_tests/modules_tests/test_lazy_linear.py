import torch
from torch import nn

from pytorch_pfn_extras.nn import LazyLinear

from pytorch_pfn_extras_tests.nn_tests.modules_tests.test_lazy import \
    LazyTestBase


class TestLazyLinear(LazyTestBase):

    def get_original_module(self):
        return nn.Linear(10, 20)

    def get_lazy_module(self):
        return LazyLinear(None, 20)

    def get_input(self):
        return torch.rand(20, 10)
