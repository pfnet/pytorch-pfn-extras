import torch

from pytorch_pfn_extras.backend import _backend_dispatcher
from pytorch_pfn_extras import engine
from pytorch_pfn_extras import reporting


class Evaluator(engine.Inferer):
    def __init__(
        self,
        backend,
        models,
        extensions=[],
        metrics=[],
        to_report_outputs=[],
        outs_fn=None,
        run_fn=None,
        device_options=None,
    ):
        super().__init__(
            backend, models, extensions, device_options=device_options
        )
        self._run_fn = backend.validation_step
        self._process_outputs_fn = backend.process_validation_outputs
        self.to_report_outputs = to_report_outputs
        # Wrap the model together with a loss fn if its specified

    # Abstract to common class?
    def get_to_report_outputs(self):
        return self.to_report_outputs

    def run(self, data):
        summary = reporting.DictSummary()
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
                    # Get the reported values and add them to the summary
                    summary.add(self._manager.observation)
        reporting.report(summary.compute_mean())


def create_evaluator(device, *args, **kwargs):
    # Get the backend
    backend = _backend_dispatcher.dispatch_backend(device)
    return Evaluator(backend, *args, **kwargs)
