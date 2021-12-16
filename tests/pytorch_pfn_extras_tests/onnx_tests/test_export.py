import typing

import numpy as np
import pytest
import torch
from flaky import flaky

from .utils import run_model_test


def test_simple():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(20, 10)
            self._outputs = []

        def forward(self, x):
            y = self.linear(x)
            self._outputs.clear()
            self._outputs.append(y)
            return self._outputs[0]

    run_model_test(Model(), (torch.rand((20,)),))


@flaky
def test_conv():
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = torch.nn.Conv2d(1, 1, 3)

        def forward(self, x):
            return self.conv(x)

    run_model_test(Net(), (torch.rand(1, 1, 112, 112),), rtol=1e-03)


def test_symbolic_function():
    class Func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a):
            return a + 10

        @staticmethod
        def symbolic(g, a):
            return g.op(
                "Add",
                a,
                g.op("Constant", value_t=torch.tensor([10], dtype=torch.float)),
            )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return Func.apply(x) + torch.tensor([10], dtype=torch.float)

    assert hasattr(Func, "symbolic")
    run_model_test(Model(), (torch.rand((20,)),))


class AnyModel(torch.nn.Module):
    def __init__(self, fn, params):
        super(AnyModel, self).__init__()
        for name, value in params.items():
            setattr(self, name, torch.nn.parameter.Parameter(torch.tensor(value)))
        self.fn = fn

    def __call__(self, *args):
        result = self.fn(self, *args)
        return result


def test_if():
    @torch.jit.script
    def if_by_shape(x):
        if x.shape[0] == 3:
            return torch.relu(x)
        else:
            return torch.abs(x)

    run_model_test(AnyModel(lambda m, x: if_by_shape(x), {}), (torch.arange(3, dtype=torch.float),))
    run_model_test(AnyModel(lambda m, x: if_by_shape(x), {}), (torch.arange(4, dtype=torch.float),))
