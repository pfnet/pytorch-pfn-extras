import torch

import pytorch_pfn_extras as ppe


def _convert(device, args):
    return [x.to(device) if isinstance(x, torch.Tensor) else x for x in args]


class TorchBackend(ppe.backend.Backend):
    def __init__(self, name):
        super().__init__(name)
        # The name is the actual device name cpu/cuda
        self._device = name

    def setup_train(self, trainer, train_loader, device_options):
        pass

    def setup_inference(self, inferencer, inf_loader, device_options):
        pass

    def setup_evaluation(self, evaluator, val_loader, device_options):
        pass

    def pre_train_step(self, trainer):
        model = trainer.get_model("main")
        model.train()

    def train_step(self, trainer, batch_idx, batch):
        # Currently using a single model,
        # Rethink this for GAN case?
        model = trainer.get_model("main")
        y = model(*_convert(self._device, batch))
        y.backward()
        return y

    def pre_validation(self, trainer, evaluator):
        model = evaluator.get_model("main")
        model.val()

    def inference_step(self, inferencer, batch):
        model = inferencer.get_model("main")
        y = model(*_convert(self._device, batch))
        return y

    def validation_step(self, evaluator, batch):
        return self.inference_step(evaluator, batch)

    def process_train_step_outputs(self, trainer, outputs):
        pass

    def process_inference_outputs(self, inferencer, outputs):
        pass

    def process_validation_outputs(self, evaluator, outputs):
        pass


ppe.backend._backend_dispatcher.register(TorchBackend("cpu"))

ppe.backend._backend_dispatcher.register(TorchBackend("cuda"))
