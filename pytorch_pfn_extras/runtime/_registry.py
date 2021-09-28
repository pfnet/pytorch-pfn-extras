from typing import Dict, Type

import torch

from pytorch_pfn_extras.runtime._runtime import DeviceLike, BaseRuntime


class _RuntimeRegistry:
    def __init__(self, fallback_class: Type[BaseRuntime]):
        self._runtimes: Dict[str, Type[BaseRuntime]] = {}
        self._fallback_class = fallback_class

    def register(
            self,
            device_type: str,
            runtime_class: Type[BaseRuntime],
    ) -> None:
        self._runtimes[device_type] = runtime_class

    def get_runtime_class_for_device_spec(
            self, device: DeviceLike) -> Type[BaseRuntime]:
        if isinstance(device, torch.device):
            device_type = device.type
        else:
            assert isinstance(device, str)
            device_type = device.split(':')[0]
        return self._runtimes.get(device_type, self._fallback_class)
