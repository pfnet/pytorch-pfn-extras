import pytest
import torch

import pytorch_pfn_extras as ppe


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='Moving across devices requires CUDA'
)
class TestPytorchRuntime:
    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    @pytest.mark.parametrize(
        'batch', [{'x': torch.zeros(1)},
                  [torch.zeros(1)],
                  torch.zeros(1),
                  object()])
    def test_convert_batch(self, device, batch):
        rt = ppe.runtime.PyTorchRuntime(device, {})
        cbatch = rt.convert_batch(batch)
        if isinstance(cbatch, dict):
            for _, v in cbatch.items():
                assert v.device.type == device
        elif isinstance(cbatch, (list, tuple)):
            for v in cbatch:
                assert v.device.type == device
        elif isinstance(cbatch, torch.Tensor):
            assert cbatch.device.type == device
        else:
            assert cbatch is batch

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_move_module(self, device):
        rt = ppe.runtime.PyTorchRuntime(device, {})
        module = torch.nn.Linear(1, 1)
        module = rt.move_module(module)
        assert module.weight.device.type == device

    @pytest.mark.parametrize('device', ['cpu', 'cuda'])
    def test_move_tensor(self, device):
        rt = ppe.runtime.PyTorchRuntime(device, {})
        tensor = torch.zeros(10)
        tensor = rt.move_tensor(tensor)
        assert tensor.device.type == device


class DummyRuntime(ppe.runtime.BaseRuntime):
    def move_module(self, module):
        return module

    def move_tensor(self, tensor):
        return tensor


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)


class SplitModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)
        ppe.to(self.layer2, device='dummy', runtime_class=DummyRuntime)


def test_runtime_container():
    module = MyModule()
    # This is a top module, so it won't show child ones
    for _ in ppe.runtime._runtime.named_runtime_modules(module):
        pytest.fail('Never reach')


def test_split_runtime_container():
    module = SplitModule()
    for name, mod in ppe.runtime._runtime.named_runtime_modules(module):
        assert name == 'layer2'
        assert mod is module.layer2


def test_split_runtime_container_recursive():
    class MultiLevelSplitModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = SplitModule()
            self.layer2 = SplitModule()
    module = MultiLevelSplitModule()
    expected = [('layer2', module.layer1.layer2),
                ('layer2', module.layer2.layer2)]
    for expected, (name, mod) in zip(
            expected, ppe.runtime._runtime.named_runtime_modules(module)):
        assert name == expected[0]
        assert mod is expected[1]

    for _ in zip(
            expected, ppe.runtime._runtime.named_runtime_modules(
            module, recursive=False)):
        pytest.fail('Never reach')


def test_module_change_forward():
    class Module1(torch.nn.Module):
        def forward(self, input):
            raise RuntimeError('The module forward should never be executed')

    class Module2:
        def __init__(self):
            self.value = 5

        def forward(self, input):
            return torch.tensor(self.value)

    class ForwardIntercepterRuntime(ppe.runtime.BaseRuntime):
        def initialize_module(self, module, loader_or_batch):
            self.new_module = Module2()
            module.forward = self.new_module.forward
            # TODO(ecastill): also reroute state_dict ?

        def move_module(self, module):
            self.initialize_module(module, None)
            return module

    module = Module1()
    with pytest.raises(RuntimeError):
        module(None)

    ppe.to(module, device='dummy', runtime_class=ForwardIntercepterRuntime)
    assert int(module(None)) == 5


def test_map():
    class Module(torch.nn.Module):
        def output(self, x):
            return {"y": x * 2, "z": x + 1}

    module = torch.nn.Sequential(Module())
    data = [{"x": torch.ones(1)}, {"x": torch.ones(2)}]
    ppe.to(module, device="cpu")
    out = list(ppe.map(module[0].output, data))
    assert len(out) == 2
    assert set(out[0].keys()) == set(["y", "z"])
    assert torch.allclose(out[0]["y"], torch.ones(1) * 2)
    assert torch.allclose(out[0]["z"], torch.ones(1) + 1)

    out = list(ppe.map(module[0].output, data, out_keys=set(["y"])))
    assert set(out[0].keys()) == set(["y"])
