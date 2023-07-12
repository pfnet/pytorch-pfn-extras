import sys

import pytest
import pytorch_pfn_extras as ppe
import torch
from pytorch_pfn_extras import testing


class _DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x).sum() * torch.tensor(5.0)


class _DummyModuleSplit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        if x.sum() < 0:
            return self.linear(x).sum() * torch.tensor(5.0)
        else:
            return self.linear(x).sum() * torch.tensor(1.0)


@pytest.mark.skipif(
    not ppe.requires("2.0.0") or sys.platform == "win32",
    reason="torch.compile interface its only added in PyTorch>2.0 and linux",
)
def test_compile_with_optimizer():
    torch._dynamo.reset()
    x = torch.randn(10, requires_grad=True)
    torch_module = _DummyModule()
    module_initial_state = torch_module.state_dict()
    compiled_module = _DummyModule()
    compiled_module.load_state_dict(module_initial_state)

    opt = torch.optim.SGD(torch_module.parameters(), lr=0.5, momentum=0.01)
    y = torch_module(x)
    y.backward()
    opt.step()

    opt = torch.optim.SGD(compiled_module.parameters(), lr=0.5, momentum=0.01)
    joint_module = ppe.compile(compiled_module, opt)
    # This executes forward+backward+optimizer step
    compiled_y = joint_module(x)
    assert torch.allclose(y, compiled_y)
    assert testing._compare_states(
        torch_module.state_dict(), compiled_module.state_dict()
    )
    # Run one more step and check that the weights now difer
    compiled_y = joint_module(x)
    assert not testing._compare_states(
        torch_module.state_dict(), compiled_module.state_dict()
    )


@pytest.mark.skipif(
    not ppe.requires("2.0.0") or sys.platform == "win32",
    reason="torch.compile interface its only added in PyTorch>2.0 and linux",
)
def test_compile_without_optimizer():
    torch._dynamo.reset()
    x = torch.randn(10, requires_grad=True)
    torch_module = _DummyModule()
    module_initial_state = torch_module.state_dict()
    compiled_module = _DummyModule()
    compiled_module.load_state_dict(module_initial_state)

    y = torch_module(x)
    y.backward()

    joint_module = ppe.compile(compiled_module, None)
    compiled_y = joint_module(x)
    # Call backward so the dummy graph is executed and the gradients are set
    # To all the tensors
    compiled_y.backward()
    assert torch.allclose(y, compiled_y)
    assert torch.allclose(
        torch_module.linear.weight.grad, compiled_module.linear.weight.grad
    )
    assert torch.allclose(
        torch_module.linear.bias.grad, compiled_module.linear.bias.grad
    )


@pytest.mark.skipif(
    not ppe.requires("2.0.0") or sys.platform == "win32",
    reason="torch.compile interface its only added in PyTorch>2.0 and linux",
)
def test_compile_with_optimizer_and_split_graph():
    torch._dynamo.reset()
    x = torch.randn(10, requires_grad=True)
    torch_module = _DummyModuleSplit()
    module_initial_state = torch_module.state_dict()
    compiled_module = _DummyModuleSplit()
    compiled_module.load_state_dict(module_initial_state)

    opt = torch.optim.SGD(torch_module.parameters(), lr=0.5, momentum=0.01)
    y = torch_module(x)
    y.backward()
    opt.step()

    opt = torch.optim.SGD(compiled_module.parameters(), lr=0.5, momentum=0.01)
    joint_module = ppe.compile(compiled_module, opt)
    # This executes forward+backward+optimizer step
    with pytest.raises(torch._dynamo.exc.Unsupported):
        joint_module(x)
