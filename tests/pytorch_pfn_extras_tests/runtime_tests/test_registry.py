import pytorch_pfn_extras as ppe
import torch


class FallbackRuntime(ppe.runtime.BaseRuntime):
    pass


class MyCustomRuntime(ppe.runtime.PyTorchRuntime):
    pass


def test_registry_register():
    registry = ppe.runtime._registry._RuntimeRegistry(FallbackRuntime)
    registry.register("dummy_device", MyCustomRuntime)
    assert (
        registry.get_runtime_class_for_device_spec("dummy_device")
        == MyCustomRuntime
    )


def test_registry_fallback():
    registry = ppe.runtime._registry._RuntimeRegistry(FallbackRuntime)
    registry.register("dummy_device", MyCustomRuntime)
    assert (
        registry.get_runtime_class_for_device_spec("unknown_device")
        == FallbackRuntime
    )


def test_registry_torch_device():
    registry = ppe.runtime._registry._RuntimeRegistry(FallbackRuntime)
    registry.register("cpu", MyCustomRuntime)
    assert (
        registry.get_runtime_class_for_device_spec(torch.device("cpu"))
        == MyCustomRuntime
    )
