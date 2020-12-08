import torch


class Backend:
    def __init__(self, name):
        self.name = name

    def setup_train(self, trainer, train_loader, device_options):
        pass

    def setup_inference(self, inferencer, inf_loader, device_options):
        pass

    def setup_evaluation(self, evaluator, val_loader, device_options):
        pass

    def pre_train_step(self, trainer):
        pass

    def train_step(self, trainer, batch_idx, batch):
        pass

    def pre_validation(self, trainer, evaluator):
        pass

    def inference_step(self, inferencer, batch):
        pass

    def validation_step(self, evaluator, batch):
        pass

    def process_train_step_outputs(self, trainer, outputs):
        pass

    def process_inference_outputs(self, inferencer, outputs):
        pass

    def process_validation_outputs(self, evaluator, outputs):
        pass

    def move_to_device(self, module, input):
        pass


class _BackendDispatcher:
    def __init__(self):
        self._backends = {}

    def register(self, backend):
        # assert isinstance(backend, Backend)
        self._backends[backend.name] = backend

    def backend_names(self):
        return list(self._backends.keys())

    def dispatch_backend(self, device):
        # Move the backend check to the actual backend
        if isinstance(device, torch.device):
            device = device.type
        return self._backends[device]


_backend_dispatcher = _BackendDispatcher()
