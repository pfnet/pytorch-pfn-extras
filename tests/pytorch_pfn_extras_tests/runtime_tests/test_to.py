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


class MyRuntime(ppe.runtime.BaseRuntime):
    def move_module(self, module):
        # Don't do the actual move
        return module

    def initialize_module(self, module, loader_or_batch):
        pass


def test_module_split_ppe_to():
    module = MyModule()
    ppe.to(module.layer2, 'dummy', runtime_class=MyRuntime,
           options={'opt': 1})
    rt_layer1 = ppe.runtime._runtime._module_runtime_tag(module.layer1)
    rt_layer2 = ppe.runtime._runtime._module_runtime_tag(module.layer2)
    assert str(next(iter(module.layer1.parameters())).device) == "cpu"
    assert rt_layer1 is None
    assert isinstance(rt_layer2, MyRuntime)
    assert rt_layer2.device_spec == 'dummy'
    assert rt_layer2.options['opt'] == 1


def test_module_split_ppe_to_config():
    # Deprecated "config" option.
    module = MyModule()
    ppe.to(module, 'dummy', runtime_class=MyRuntime, config={'opt': 1})
    rt_layer1 = ppe.runtime._runtime._module_runtime_tag(module)
    assert isinstance(rt_layer1, MyRuntime)
    assert rt_layer1.options['opt'] == 1
