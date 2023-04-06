import contextlib
import dataclasses
from typing import Any, Dict, Generator, Iterable, Mapping, Optional
import warnings

import torch

from pytorch_pfn_extras.handler._code_block import forward, update_parameters
from pytorch_pfn_extras.runtime import _autocast


# Deprecated: kept for backward compatibility of user code
@contextlib.contextmanager
def torch_autocast(enabled: bool = True) -> Generator[None, None, None]:
    if _autocast._cuda_amp_available:
        with torch.cuda.amp.autocast(enabled):  # type: ignore[no-untyped-call]
            yield
    else:
        yield


def _normalize_outputs(outputs: Any) -> Dict[str, Any]:
    target: Dict[str, Any]
    if isinstance(outputs, tuple) and hasattr(outputs, '_fields'):
        # namedtuple
        target = outputs._asdict()  # type: ignore[attr-defined]
    elif isinstance(outputs, dict):
        target = outputs
    elif isinstance(outputs, (list, tuple)):
        target = {str(i): out for i, out in enumerate(outputs)}
    else:
        target = {"0": outputs}
    return target


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
            models: Mapping[str, torch.nn.Module],
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
            models: Mapping[str, torch.nn.Module],
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
            models: Mapping[str, torch.nn.Module],
            optimizers: Mapping[str, torch.optim.Optimizer],
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
            models: Mapping[str, torch.nn.Module],
            optimizers: Mapping[str, torch.optim.Optimizer],
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
            models: Mapping[str, torch.nn.Module]
    ) -> None:
        """A method called when starting a validation.

        Args:
            models (dict of torch.nn.Module): The models.
        """
        pass

    def train_validation_end(
            self,
            models: Mapping[str, torch.nn.Module],
    ) -> None:
        """A method called when the validation completes.

        Args:
            models (dict of torch.nn.Module): The models.
        """
        pass

    def eval_step(
            self,
            models: Mapping[str, torch.nn.Module],
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
                * ``'autocast'`` (bool or dict):
                    If ``True``, ``torch.autocast`` (or ``torch.cuda.amp.autocast`` for PyTorch 1.9 or earlier) is enabled,
                    using ``{"enabled": True, "device_type": "cuda"}``
                    as autocast options.
                    The default is ``False`` which corresponds to the following options
                    ``{"enabled": False, "device_type": "cuda"}``.
                    If dict, options are passed to ``torch.autocast``.
                * ``'grad_scaler'`` (torch.cuda.amp.GradScaler):
                    A gradient scaler that outputs are applied to.
        """
        super().__init__(options)
        self.model_name = model_name

    def consume_options(self, options: Dict[str, Any]) -> None:
        super().consume_options(options)

        self.backward_outputs = options.pop('backward_outputs', None)
        self._grad_scaler = options.pop('grad_scaler', None)

        self._backward_fn = options.pop('backward_function', None)
        autocast_options = options.get("autocast", False)
        if isinstance(autocast_options, bool):
            autocast_options = {"enabled": autocast_options, "device_type": "cuda"}
        self._autocast = _autocast._AutocastManager(
            autocast_options, self._grad_scaler is not None
        )

        if self._grad_scaler is not None:
            if not isinstance(self._grad_scaler, torch.cuda.amp.GradScaler):
                raise RuntimeError('grad_scaler should be a '
                                   'torch.cuda.amp.GradScaler object')

    def _forward(self, model: torch.nn.Module, batch: Any) -> Any:
        if isinstance(batch, tuple) and hasattr(batch, '_fields'):
            # namedtuple
            return model(batch)
        if isinstance(batch, dict):
            return model(**batch)
        if isinstance(batch, (list, tuple)):
            return model(*batch)
        return model(batch)

    def _backward(self, outputs: Dict[str, Any]) -> None:
        to_backward = set()
        if self.backward_outputs is None:
            for _, v in outputs.items():
                if isinstance(v, torch.Tensor) and v.grad_fn is not None and (
                    (
                        v.numel() == 1
                        and (v.dtype.is_floating_point or v.dtype.is_complex)
                    )
                ):
                    to_backward.add(v)
        else:
            # If backward is requested, we tried to execute it no matter the
            # shape or type of the tensor to make the user aware
            backward_outputs = self.backward_outputs
            if type(backward_outputs) is str:
                backward_outputs = (backward_outputs,)
            for k in backward_outputs:
                try:
                    v = outputs[k]
                    if isinstance(v, torch.Tensor) and v.grad_fn is not None:
                        to_backward.add(v)
                except KeyError:
                    warnings.warn(
                        'Couldn\'t find requested backward value: '
                        f'{k} in {outputs.keys()}'
                    )

        for v in to_backward:
            if self._backward_fn is None:
                v.backward()  # type: ignore[no-untyped-call]
            else:
                self._backward_fn(v)

    def train_epoch_begin(
            self,
            models: Mapping[str, torch.nn.Module],
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
            models: Mapping[str, torch.nn.Module],
            optimizers: Mapping[str, torch.optim.Optimizer],
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
        with self._autocast.autocast():
            optimizers[self.model_name].zero_grad()
            outs = self._forward(models[self.model_name], batch)
            to_back_outs = _normalize_outputs(outs)
            if self._grad_scaler is not None:
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
            models: Mapping[str, torch.nn.Module],
            optimizers: Mapping[str, torch.optim.Optimizer],
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
            models: Mapping[str, torch.nn.Module],
    ) -> None:
        """A method called when starting a validation.

        Args:
            models (dict of torch.nn.Module): The models.
        """
        model = models[self.model_name]
        model.eval()

    def eval_step(
            self,
            models: Mapping[str, torch.nn.Module],
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


class CodeBlockLogic(BaseLogic):
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
        """
        super().__init__(options)
        self.model_name = model_name

    def consume_options(self, options: Dict[str, Any]) -> None:
        super().consume_options(options)

        self.backward_outputs = options.pop('backward_outputs', None)
        if self.backward_outputs is not None:
            assert isinstance(self.backward_outputs, str)

    def train_epoch_begin(
            self,
            models: Mapping[str, torch.nn.Module],
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
            models: Mapping[str, torch.nn.Module],
            optimizers: Mapping[str, torch.optim.Optimizer],
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
        module = models[self.model_name]

        return update_parameters(
            module,
            list(optimizers.values()),
            self.backward_outputs,
            None,
        )(batch)

    def train_validation_begin(
            self,
            models: Mapping[str, torch.nn.Module],
    ) -> None:
        """A method called when starting a validation.

        Args:
            models (dict of torch.nn.Module): The models.
        """
        model = models[self.model_name]
        model.eval()

    def eval_step(
            self,
            models: Mapping[str, torch.nn.Module],
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
        outs = forward(model)(batch)
        return outs


@dataclasses.dataclass
class ClousureModelOutput:
    outs: Any
    loss: torch.Tensor

    def __float__(self) -> float:
        return float(self.loss)


class ClousureLogic(Logic):

    def consume_options(self, options: Dict[str, Any]) -> None:
        super().consume_options(options)
        if self._grad_scaler is not None:
            raise RuntimeError('torch.cuda.amp.GradScaler does not support clousure step mode.')

    def train_step(
            self,
            models: Mapping[str, torch.nn.Module],
            optimizers: Mapping[str, torch.optim.Optimizer],
            batch_idx: int,
            batch: Any,
    ) -> Any:
        """A method invokes the model forward and backward passes and performs an optimization step.

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
        def clousure() -> ClousureModelOutput:
            with self._autocast.autocast():
                optimizers[self.model_name].zero_grad()
                outs = self._forward(models[self.model_name], batch)
            to_back_outs = _normalize_outputs(outs)
            if len(to_back_outs) > 1:
                raise RuntimeError("Clousure step with multiple outputs is not supported.")
            elif len(to_back_outs) == 0:
                raise RuntimeError("No backward target found.")

            self._backward(to_back_outs)
            loss, = to_back_outs.values()
            return ClousureModelOutput(
                outs=outs,
                loss=loss,
            )

        optimizer = optimizers[self.model_name]
        clousure_model_output: ClousureModelOutput = optimizer.step(clousure)  # type: ignore
        if not isinstance(clousure_model_output, ClousureModelOutput):
            raise RuntimeError(f"{type(clousure_model_output)} type object returned from optimizer.step with clousure. optimizer.step is expected to return ppe.handler.ClousureModelOutput.")
        return clousure_model_output.outs

    def train_step_optimizers(
            self,
            models: Mapping[str, torch.nn.Module],
            optimizers: Mapping[str, torch.optim.Optimizer],
            batch_idx: int,
    ) -> None:
        """In clousure mode, the stepping of the optimizer cannot be changed.

        If you want to change the stepping of the optimizer, please use the normal Logic class.

        Args:
            optimizers (dict of torch.optim.Optimizer):
                The optimizers.
            batch_idx (int):
                Number of steps already finished.
        """
        pass
