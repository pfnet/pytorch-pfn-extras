import torch

import pytorch_pfn_extras as ppe


def move_to(module, device_name):
    backend_name = device_name.split(":")[0]
    backend = ppe.backend._backend_dispatcher._backends[backend_name]
    return backend.move_func(module, device_name)


class _Engine:
    def __init__(self, *args, device_options=None, **kwargs):
        self._device_options = device_options
        self._manager = ppe.training.ExtensionsManager(*args, **kwargs)

    def extend(
        self,
        extension,
        name=None,
        trigger=None,
        priority=None,
        *,
        call_before_training=False,
        **kwargs,
    ):
        self._manager.extend(
            extension,
            name,
            trigger,
            priority,
            call_before_training=call_before_training,
            **kwargs,
        )

    def get_extension(self, name):
        self._manager.get_extension(name)

    def state_dict(self, *, transform_models=lambda n, x: x):
        return self._manager.state_dict(transform_models=transform_models)

    def load_state_dict(self, to_load, *, transform_models=lambda n, x: x):
        self._manager.load_state_dict(
            to_load, transform_models=transform_models
        )

    @property
    def models(self):
        return self._manager.models

    def get_model(self, name):
        return self.models[name]

    def set_model(self, name, model):
        self.models[name] = model

    def run(self, train_loader):
        raise NotImplementedError


class Inferer(_Engine):
    def __init__(
        self,
        backend,
        models,
        extensions=[],
        outs_fn=None,
        run_fn=None,
        device_options=None,
    ):
        super().__init__(
            models, {}, 1, iters_per_epoch=1, extensions=extensions,
        )
        self._run_fn = run_fn
        self._process_outputs_fn = outs_fn

    def run(self, data):
        self.models["main"].eval()
        # For extensions such as the progress bar to work
        self._manager._iters_per_epoch = len(data)
        with torch.no_grad():
            for i, x in enumerate(data):
                with self._manager.run_iteration():
                    outs = self._run_fn(self, x)
                    # TODO: Make run like a generator that yields
                    # the output to the user code instead of adding
                    # this callback like?
                    self._process_outputs_fn(self, outs)


def create_inferer(device, *args, **kwargs):
    # Get the backend
    backend = ppe.backend._backend_dispatcher.dispatch_backend(device)
    return ppe.engine.Inferer(backend, *args, **kwargs)
