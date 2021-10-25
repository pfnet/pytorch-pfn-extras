from typing import Any, Dict, Optional, Type, TypeVar

import torch

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.runtime._runtime import DeviceLike, BaseRuntime


ModuleOrTensor = TypeVar('ModuleOrTensor', torch.nn.Module, torch.Tensor)


def to(
        module_or_tensor: ModuleOrTensor,
        device: DeviceLike,
        *,
        config: Optional[Dict[str, Any]] = None,
        runtime_class: Optional[Type[BaseRuntime]] = None,
) -> ModuleOrTensor:
    """A function to transfer the given object to the given device.

    If PyTorch's device type is given as the ``device`` argument,
    the behavior of this function is  equivalent to
    ``module_or_tensor.to(module_or_tensor, device)``.

    Otherwise, this function uses the **Runtime** mechanism.
    This function looks for the Runtime for the device from the RuntimeRegistry
    and delegates the actual transfer operation to it.

    See also the documentation of ``ppe.runtime.BaseRuntime`` for details.

    Args:
        module_or_tensor (torch.nn.Module or torch.Tensor):
            An object to be transferred.
        device (torch.device or str):
            The device that the input object is transferred to.
        config (dict, optional):
            A config of dictionary type that is passed to
            ``runtime_class.__init__`` as an argument.
        runtime_class:
            A runtime class inherited from `BaseRuntime` class.
            If ``None``, a runtime class is automatically selected
            based on the ``device`` argument from the runtime registry.

    Returns:
        A `torch.Tensor` with the specified device.
    """
    if config is None:
        config = {}
    if runtime_class is None:
        registry = ppe.runtime.runtime_registry
        runtime_class = registry.get_runtime_class_for_device_spec(device)
    runtime = runtime_class(device, config)
    obj = module_or_tensor
    if isinstance(obj, torch.nn.Module):
        ppe.runtime._runtime._set_module_runtime_tag(obj, runtime)
        return runtime.move_module(obj)
    elif isinstance(obj, torch.Tensor):
        return runtime.move_tensor(obj)
    else:
        raise ValueError('Unsupported type for module_or_tensor')
