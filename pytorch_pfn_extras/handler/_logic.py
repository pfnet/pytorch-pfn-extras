import contextlib
from typing import (
    Any, Dict, Generator, Iterable, Optional,
)

_amp_enabled = False


try:
    import torch.cuda.amp
    _amp_enabled = torch.cuda.is_available() and hasattr(
        torch.cuda.amp, 'autocast')
except ImportError:
    pass


@contextlib.contextmanager
def torch_autocast(enabled: bool = True) -> Generator[None, None, None]:
    if _amp_enabled:
        with torch.cuda.amp.autocast(enabled):  # type: ignore[no-untyped-call]
            yield
    else:
        yield


class BaseLogic:
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__()
        options = options.copy() if options else {}
        self.consume_options(options)

    def consume_options(self, options: Dict[str, Any]) -> None:
        """A method to update options of Logic.

        Note that the given dict will be modified.

        Args:
            options (dict): Option key-values to be set.
        """
        pass

    def train_epoch_begin(
            self,
            models: Dict[str, torch.nn.Module],
            epoch: int,
            loader: Iterable[Any],
    ) -> None:
        """A method called when starting a new epoch of training.

        Args:
            epoch (int): Number of epochs already finished.
            models (dict of torch.nn.Module): The models.
            loader (torch.utils.data.DataLoader): The data loder.
        """
        pass

    def train_epoch_end(
            self,
            models: Dict[str, torch.nn.Module],
            epoch: int,
    ) -> None:
        """A method called when completing an epoch of training.

        Args:
            epoch (int): Number of epochs already finished.
            models (dict of torch.nn.Module): The models.
        """
        pass

    def train_step(
            self,
            models: Dict[str, torch.nn.Module],
            optimizers: Dict[str, torch.optim.Optimizer],
            batch_idx: int,
            batch: Any,
    ) -> Any:
        """A method invokes the models forward and backward passes.

        Optimizing is left to `train_step_optimizers` since maybe the user
        would like to aggregate the gradients of several iterations.

        Args:
            models (dict of torch.nn.Module):
                The models.
            optimizers (dict of torch.optim.Optimizer):
                The optimizers.
            batch_idx (int):
                Number of training steps already finished.
            batch (torch.Tensor, list of torch.Tensor, dict of torch.Tensor):
                Input tensors feeded to the model of the current step.
        """
        pass

    def train_step_optimizers(
            self,
            models: Dict[str, torch.nn.Module],
            optimizers: Dict[str, torch.optim.Optimizer],
            batch_idx: int,
    ) -> None:
        """A method in charge of stepping the provided optimizers.

        Args:
            optimizers (dict of torch.optim.Optimizer):
                The optimizers.
            batch_idx (int):
                Number of steps already finished.
        """
        pass

    def train_validation_begin(
            self,
            models: Dict[str, torch.nn.Module]
    ) -> None:
        """A method called when starting a validation.

        Args:
            models (dict of torch.nn.Module): The models.
        """
        pass

    def train_validation_end(
            self,
            models: Dict[str, torch.nn.Module],
    ) -> None:
        """A method called when the validation completes.

        Args:
            models (dict of torch.nn.Module): The models.
        """
        pass

    def eval_step(
            self,
            models: Dict[str, torch.nn.Module],
            batch_idx: int,
            batch: Any,
    ) -> Any:
        """A method for an evaluation step.

        Args:
            models (dict of torch.nn.Module): The models.
            batch_idx (int): Number of steps already finished.
            batch (torch.Tensor, list of torch.Tensor, dict of torch.Tensor):
                Input tensors feeded to the model of the current step.
        """
        pass


class Logic(BaseLogic):

    def __init__(
            self,
            model_name: str = 'main',
            options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """A set of methods that defines the training logic.

        Args:
            model_name (str): Name of the model. Default is ``'main'``.
            options (dict, optional): The configuration options.

                * ``'backward_outputs'`` (list of str):
                    A list of names of outputs that require compution of
                    the gradient.
                * ``'autocast'`` (bool):
                    If ``True``, ``torch.cuda.amp.autocast`` is enabled.
                    Default is ``False``.
                * ``'grad_scaler'`` (torch.cuda.amp.GradScaler):
                    A gradient scaler that outputs are applied to.
        """
        super().__init__(options)
        self.model_name = model_name

    def consume_options(self, options: Dict[str, Any]) -> None:
        super().consume_options(options)

        self.backward_outputs = options.pop('backward_outputs', None)
        self._grad_scaler = options.pop('grad_scaler', None)
        self._autocast = options.pop('autocast', False)

        if not _amp_enabled:
            if self._grad_scaler is not None or self._autocast:
                raise RuntimeError('Requested AMP features but torch.cuda.amp'
                                   ' is not enabled')

        if self._grad_scaler is not None:
            if not isinstance(self._grad_scaler, torch.cuda.amp.GradScaler):
                raise RuntimeError('grad_scaler should be a '
                                   'torch.cuda.amp.GradScaler object')

    def _forward(self, model: torch.nn.Module, batch: Any) -> Any:
        if isinstance(batch, tuple) and hasattr(batch, '_fields'):
            return model(batch)
        if isinstance(batch, dict):
            return model(**batch)
        if isinstance(batch, (list, tuple)):
            return model(*batch)
        return model(batch)

    def _normalize_outputs(self, outputs: Any) -> Dict[str, Any]:
        if isinstance(outputs, tuple) and hasattr(outputs, '_fields'):
            fields = outputs._fields  # type: ignore[attr-defined]
            target = {k: getattr(outputs, k) for k in fields}
        elif isinstance(outputs, dict):
            target = outputs
        elif isinstance(outputs, (list, tuple)):
            target = {str(i): out for i, out in enumerate(outputs)}
        else:
            target = {"0": outputs}
        return target

    def _backward(self, outputs: Dict[str, Any]) -> None:
        target = self._normalize_outputs(outputs)

        for k, v in target.items():
            # This is to avoid errors when the trained models returns
            # tensors others than scalars
            if isinstance(v, torch.Tensor) and v.grad_fn is not None and (
                (
                    self.backward_outputs is None
                    and v.numel() == 1
                    and (v.dtype.is_floating_point or v.dtype.is_complex)
                )
                or (
                    self.backward_outputs is not None
                    and k in self.backward_outputs
                )
            ):
                v.backward()  # type: ignore[no-untyped-call]

    def train_epoch_begin(
            self,
            models: Dict[str, torch.nn.Module],
            epoch: int,
            loader: Iterable[Any],
    ) -> None:
        """A method called when starting a new epoch of training.

        Args:
            epoch (int): Number of epochs already finished.
            models (dict of torch.nn.Module): The models.
            loader (torch.utils.data.DataLoader): The data loder.
        """
        model = models[self.model_name]
        model.train()
        if hasattr(loader, 'sampler') and hasattr(
                loader.sampler, 'set_epoch'):  # type: ignore[attr-defined]
            # Needed for `torch.utils.data.DistributedSampler`
            loader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]

    def train_step(
            self,
            models: Dict[str, torch.nn.Module],
            optimizers: Dict[str, torch.optim.Optimizer],
            batch_idx: int,
            batch: Any,
    ) -> Any:
        """A method invokes the model forward and backward passes.

        Optimizing is left to `train_step_optimizers` since maybe the user
        would like to aggregate the gradients of several iterations.

        Args:
            models (dict of torch.nn.Module):
                The models.
            optimizers (dict of torch.optim.Optimizer):
                The optimizers.
            batch_idx (int):
                Number of training steps already finished.
            batch (torch.Tensor, list of torch.Tensor, dict of torch.Tensor):
                Input tensors feeded to the model of the current step.
        """
        with torch_autocast(enabled=self._autocast):
            optimizers[self.model_name].zero_grad()
            outs = self._forward(models[self.model_name], batch)
            to_back_outs = outs
            if self._grad_scaler is not None:
                to_back_outs = self._normalize_outputs(outs)
                assert (
                    len(to_back_outs) == 1
                ), "loss scaling with multiple outputs is not supported"
                to_back_outs = {
                    k: self._grad_scaler.scale(v)
                    for k, v in to_back_outs.items()}
        self._backward(to_back_outs)
        return outs

    def train_step_optimizers(
            self,
            models: Dict[str, torch.nn.Module],
            optimizers: Dict[str, torch.optim.Optimizer],
            batch_idx: int,
    ) -> None:
        """A method in charge of stepping the provided optimizers.

        Also a grad scaler will be used if defined.

        Args:
            optimizers (dict of torch.optim.Optimizer):
                The optimizers.
            batch_idx (int):
                Number of steps already finished.
        """
        optimizer = optimizers[self.model_name]
        if self._grad_scaler is not None:
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()
        else:
            optimizer.step()

    def train_validation_begin(
            self,
            models: Dict[str, torch.nn.Module],
    ) -> None:
        """A method called when starting a validation.

        Args:
            models (dict of torch.nn.Module): The models.
        """
        model = models[self.model_name]
        model.eval()

    def eval_step(
            self,
            models: Dict[str, torch.nn.Module],
            batch_idx: int,
            batch: Any,
    ) -> Any:
        """A method for an evaluation step.

        Args:
            models (dict of torch.nn.Module): The models.
            batch_idx (int): Number of steps already finished.
            batch (torch.Tensor, list of torch.Tensor, dict of torch.Tensor):
                Input tensors feeded to the model of the current step.
        """
        model = models[self.model_name]
        outs = self._forward(model, batch)
        return outs
