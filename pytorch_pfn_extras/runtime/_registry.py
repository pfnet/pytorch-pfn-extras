import torch


class _RuntimeRegistry:
    def __init__(self, fallback_class):
        self._runtimes = {}
        self._fallback_class = fallback_class

    def register(self, device_type, runtime_class):
        self._runtimes[device_type] = runtime_class

    def get_runtime_class_for_device_spec(self, device):
        if isinstance(device, torch.device):
            device_type = device.type
        else:
            assert isinstance(device, str)
            device_type = device.split(':')[0]
        return self._runtimes.get(device_type, self._fallback_class)
