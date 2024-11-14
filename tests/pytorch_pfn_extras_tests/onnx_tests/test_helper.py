import pytest
import pytorch_pfn_extras
import torch
import torch.nn as nn
import torch.onnx

from pytorch_pfn_extras.onnx import no_grad


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_no_grad_no_export():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.linear = nn.Linear(30, 20, bias=False)

        def forward(self, x):
            h = no_grad(lambda : self.linear(self.conv(x)))
            return h

    model = Net()
    x = torch.ones((1, 1, 32, 32))
    y = model(x)
    assert not y.requires_grad


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_no_grad():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.linear = nn.Linear(30, 20, bias=False)

        def forward(self, x):
            h = no_grad(lambda : self.linear(self.conv(x)))
            return h
    model = Net()
    x = torch.ones((1, 1, 32, 32))

    # use torch.jit because `torch.onnx` removes detach op by default.
    graph = torch.jit.trace(model, (x,))
    last_node = list(graph.graph.nodes())[-1]
    assert last_node.kind() == "aten::detach"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_no_grad_with_list_output():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.linear = nn.Linear(30, 20, bias=False)

        def forward(self, x):
            def f(x):
                h = self.conv(x)
                return [h, self.linear(h)]
            h0, h1 = no_grad(f, x)
            return h0.sum() + h1.sum()
    model = Net()
    x = torch.ones((1, 1, 32, 32))

    # use torch.jit because `torch.onnx` removes detach op by default.
    graph = torch.jit.trace(model, (x,))
    detach_nodes = [x for x in graph.graph.nodes() if x.kind() == "aten::detach"]
    assert len(detach_nodes) == 2


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_no_grad_with_dict_output():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(1, 6, 3)
            self.linear = nn.Linear(30, 20, bias=False)

        def forward(self, x):
            def f(x):
                h = self.conv(x)
                return {"h0": h, "h1": self.linear(h)}
            h0, h1 = no_grad(f, x).values()
            return h0.sum() + h1.sum()
    model = Net()
    x = torch.ones((1, 1, 32, 32))

    # use torch.jit because `torch.onnx` removes detach op by default.
    graph = torch.jit.trace(model, (x,))
    detach_nodes = [x for x in graph.graph.nodes() if x.kind() == "aten::detach"]
    assert len(detach_nodes) == 2
