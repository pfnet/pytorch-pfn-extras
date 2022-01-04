import pytest
import torch

import pytorch_pfn_extras as ppe


@pytest.mark.gpu
def test_tensor_ppe_to():
    device = 'cuda:0'
    tensor = torch.zeros(10)
    out = ppe.to(tensor, device)
    assert str(out.device) == device


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)


@pytest.mark.gpu
def test_module_ppe_to():
    device = 'cuda:0'
    module = MyModule()
    ppe.to(module, device)
    assert all([str(p.device) == device for p in module.parameters()])
    assert ppe.runtime._runtime._module_runtime_tag(module) is not None


def test_invalid_ppe_to():
    device = 'cpu'
    with pytest.raises(ValueError):
        ppe.to(object(), device)


@pytest.mark.gpu
def test_module_split_ppe_to():
    class TestRuntime(ppe.runtime.BaseRuntime):
        def move_module(self, module):
            # Don't do the actual move
            return module

        def initialize_module(self, module, loader_or_batch):
            pass

    module = MyModule()
    ppe.to(module.layer2, 'dummy', runtime_class=TestRuntime)
    assert str(next(iter(module.layer1.parameters())).device) == "cpu"
    assert ppe.runtime._runtime._module_runtime_tag(module.layer1) is None
    assert ppe.runtime._runtime._module_runtime_tag(module.layer2) is not None


def test_runtime_nested():
    class TestRuntime(ppe.runtime.BaseRuntime):
        def move_module(self, module):
            # Don't do the actual move
            return module

        def initialize_module(self, module, loader_or_batch):
            pass

    module = MyModule()
    ppe.to(module, 'dummy', runtime_class=TestRuntime)
    ppe.to(module.layer2, 'dummy', runtime_class=TestRuntime)
    with pytest.raises(ValueError, match="nested"):
        for _ in ppe.runtime._runtime.named_runtime_modules(module):
            pass
