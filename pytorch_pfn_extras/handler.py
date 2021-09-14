from collections import defaultdict
import contextlib
from typing import (
    Any, Callable, Dict, Generator, Iterable, List, Optional,
    Tuple, Union, TYPE_CHECKING,
)

import torch

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras import reporting

if TYPE_CHECKING:
    from pytorch_pfn_extras.training._trainer import _Trainer
    from pytorch_pfn_extras.training._evaluator import _Evaluator
    from pytorch_pfn_extras.runtime import BaseRuntime


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
                    len(outs) == 1
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


class BaseHandler:

    def __init__(
            self,
            logic: BaseLogic,
            options: Dict[str, Any],
            *args: Any,
            **kwargs: Any
    ) -> None:
        """Base class of Handler.

        .. seealso:
           :class:`pytorch_pfn_extras.handler.Handler`

        Args:
            logic (Logic): A logic.
        """
        super().__init__()
        options = options.copy() if options else {}
        self._logic = logic
        self.consume_options(options)

    def consume_options(self, options: Dict[str, Any]) -> None:
        """A method to update options of Handler.

        Note that the given dict will be modified.

        Args:
            options (dict): Option key-values to be set.

        .. seealso:
           :meth:`pytorch_pfn_extras.handler.Handler.consume_options`
        """
        pass

    def train_setup(self, trainer: '_Trainer', loader: Iterable[Any]) -> None:
        """A method called only once when starting a training run.

        .. seealso:
           :meth:`pytorch_pfn_extras.handler.Handler.train_step`
        """
        # Context: Trainer
        # Called only once when starting a training run.
        pass

    def train_epoch_begin(
            self,
            trainer: '_Trainer',
            loader: Iterable[Any]
    ) -> None:
        """A method called when starting a new epoch.

        .. seealso:
           :meth:`pytorch_pfn_extras.handler.Handler.train_epoch_begin`
        """
        # Context: Trainer
        # Called when starting a new epoch.
        pass

    def train_epoch_end(self, trainer: '_Trainer') -> None:
        """A method called when finishing an epoch.

        .. seealso:
           :meth:`pytorch_pfn_extras.handler.Handler.train_epoch_end`
        """
        # Context: Trainer
        # Called when finishing an epoch.
        pass

    def train_validation_begin(
            self,
            trainer: '_Trainer',
            evaluator: '_Evaluator',
    ) -> None:
        """A method called when starting a validation.

        .. seealso:
           :meth:`pytorch_pfn_extras.handler.Handler.train_validation_begin`
        """
        # Context: Trainer
        # Called just before starting a validation run, i.e. at the end of
        # every epoch in the training run.
        pass

    def train_validation_end(
            self,
            trainer: '_Trainer',
            evaluator: '_Evaluator',
    ) -> None:
        """A method called after validation.

        Args:
            trainer (Trainer): The trainer that calls this method.
            evaluator (Evaluator): The evaluator used for validation.
        """
        # Context: Trainer
        # Called after validation run, i.e. at the end of
        # every epoch in the training run.
        pass

    def train_step(
            self,
            trainer: '_Trainer',
            batch_idx: int,
            batch: Any,
            complete_fn: Callable[[int, Any], None],
    ) -> None:
        """A training step.

        .. seealso:
           :meth:`pytorch_pfn_extras.handler.Handler.train_step`
        """
        # Context: Trainer
        # Do a training iteration.
        pass

    def train_post_step(
            self,
            trainer: '_Trainer',
            batch_idx: int,
            batch: Any,
            outputs: Any,
    ) -> None:
        """A method called after each training step.

        .. seealso:
           :meth:`pytorch_pfn_extras.handler.Handler.train_post_step`
        """
        # Context: Trainer
        # Called after train_step.
        pass

    def eval_setup(
            self,
            evaluator: '_Evaluator',
            loader: Iterable[Any]
    ) -> None:
        """A method called only once when starting a training run.
        When evaluator is not given, this method is not called.

        .. seealso:
           :meth:`pytorch_pfn_extras.handler.Handler.eval_setup`
        """
        # Context: Evaluator
        # Called only once when starting a training run, when evaluator is
        # given.
        pass

    def eval_loop_begin(self, evaluator: '_Evaluator') -> None:
        """A method called before each evaluation step.

        Args:
            evaluator (Evaluator): The evaluator.
        """
        # Context: Evaluator
        # Called before running all the steps of the evaluation
        pass

    def eval_step(
            self,
            evaluator: '_Evaluator',
            batch_idx: int,
            batch: Any,
            complete_fn: Callable[[int, Any], None],
    ) -> None:
        """Evaluation iteration.

        .. seealso:
           :meth:`pytorch_pfn_extras.handler.Handler.eval_step`
        """
        # Context: Evaluator
        # Do an evaluation iteration.
        pass

    def eval_loop_end(self, evaluator: '_Evaluator') -> None:
        """A method called after running all steps of the evaluation.

        .. seealso:
           :meth:`pytorch_pfn_extras.handler.Handler.eval_loop_endp`
        """
        # Context: Evaluator
        # Called after running all the steps of the evaluation
        pass

    def eval_post_step(
            self,
            evaluator: '_Evaluator',
            batch_idx: int,
            batch: Any,
            outputs: Any,
    ) -> None:
        """A method called after each evaluation step.

        .. seealso:
           :meth:`pytorch_pfn_extras.handler.Handler.eval_post_step`
        """
        # Context: Evaluator
        # Called after eval_step.
        pass


ModulesTuple = Tuple[str, torch.nn.Module, 'BaseRuntime']
PendingIters = List[Tuple[int, Any, Callable[..., None]]]


class Handler(BaseHandler):

    def __init__(
            self,
            logic: BaseLogic,
            entry_runtime: 'BaseRuntime',
            options: Dict[str, Any],
    ) -> None:
        """A set of callback functions to perform device-specific operations.

        Args:
            logic (Logic): A logic.
            entry_runtime (BaseRuntime): A runtime object.
            options (dict): The configuration options.

                * ``'eval_report_keys'`` (list of str):
                    A list of names of outputs that are given as inputs
                    of ``reporting.report`` after each evaluation step.
                    Default is an empty list.
                * ``'train_report_keys'`` (list of str):
                    A list of names of outputs that are given as inputs
                    of ``reporting.report`` after each training step.
                    Default is an empty list.
                * ``'async'`` (bool):
                    If ``True``, async mode is enabled. Default is ``False``.
        """
        super().__init__(logic, options)
        self.pending_iters: Dict[str, PendingIters] = defaultdict(list)

        # This is used to send the batch to the appropiate device
        self._entry_runtime = entry_runtime
        self._ppe_modules: List[ModulesTuple] = []

    def consume_options(self, options: Dict[str, Any]) -> None:
        super().consume_options(options)
        self._eval_report_keys = options.pop('eval_report_keys', [])
        self._train_report_keys = options.pop('train_report_keys', [])
        self._async = options.pop('async', False)

    def _runtime_iterator(
            self,
            models: Dict[str, torch.nn.Module],
    ) -> Generator[ModulesTuple, ModulesTuple, None]:
        if not self._ppe_modules:
            for n, m in models.items():
                for sn, sm in ppe.runtime._runtime.named_runtime_modules(m, n):
                    rt = ppe.runtime._runtime._module_runtime_tag(sm)
                    self._ppe_modules.append((sn, sm, rt))
                    yield sn, sm, rt
        else:
            for sn, sm, rt in self._ppe_modules:
                yield sn, sm, rt

    def _setup(
            self,
            models: Dict[str, torch.nn.Module],
            loader: Union[Iterable[Any], Dict[str, Iterable[Any]]],
            optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    ) -> None:
        # This requires loader to be always a dict
        # should be avoided?
        if not isinstance(loader, dict):
            # The default model always has empty name when obtained from the
            # modules
            loaders = {sn: loader
                       for sn, _, _ in self._runtime_iterator(models)}
        else:
            loaders = loader
        if optimizers is None:
            optimizers = {}

        for sn, sm, rt in self._runtime_iterator(models):
            # users can give a tensor or loader in case
            # shape cannot be inferred for submodules
            # TODO the optimizers are also needed?
            optim = optimizers.get(sn, None)
            if optim is None and len(optimizers) == 1:
                optim = next(iter(optimizers.values()))
            load = loaders.get(sn, None)
            rt.initialize_module(sm, load, optim)

        # Split model can't be used with async unless
        # we just rewrite the model forward to defer part of its execution
        # until the result is available
        if self._async and len(self._ppe_modules) != 1:
            raise RuntimeError("Async mode is not supported in models "
                               "splitted across different devices")

    def train_setup(self, trainer: '_Trainer', loader: Iterable[Any]) -> None:
        """A method called only once when starting a training run.

        Args:
            trainer (Trainer): The trainer that calls this method.
            loader (torch.utils.data.DataLoader): The data loader.
        """
        for _, model in trainer.models.items():
            model.train()
        self._setup(trainer.models, loader, trainer.optimizers)

    def train_epoch_begin(
            self,
            trainer: '_Trainer',
            loader: Iterable[Any]
    ) -> None:
        """A method called when starting a new epoch.

        Args:
            trainer (Trainer): The trainer that calls this method.
            loader (torch.utils.data.DataLoader): The data loader.
        """
        for _, sm, rt in self._runtime_iterator(trainer.models):
            rt.train_epoch_begin(sm)

        self._logic.train_epoch_begin(trainer.models, trainer.epoch, loader)

    def train_epoch_end(self, trainer: '_Trainer') -> None:
        """A method called when finishing an epoch.

        Args:
            trainer (Trainer): The trainer that calls this method.
        """
        if self._async:
            while self.pending_iters:
                # TODO(ecastill) block until we get the result
                for sn, sm, rt in self._runtime_iterator(trainer.models):
                    outs = rt.get_pending_result(sm, True)
                    if outs is not None:
                        self._complete_train_step(
                            trainer, outs, True, sn, sm, rt)

        self._logic.train_epoch_end(trainer.models, trainer.epoch)

    def train_validation_begin(
            self,
            trainer: '_Trainer',
            evaluator: '_Evaluator',
    ) -> None:
        """A method called when starting a validation.

        Args:
            evaluator (Evaluator): An evaluator.
        """
        # We need to correlate the models in trainer and evaluator
        for _, sm, rt in self._runtime_iterator(evaluator.models):
            rt.train_validation_begin(sm)
        self._logic.train_validation_begin(evaluator.models)

    def train_validation_end(
            self,
            trainer: '_Trainer',
            evaluator: '_Evaluator',
    ) -> None:
        """A method called after validation.

        Args:
            trainer (Trainer): The trainer that calls this method.
            evaluator (Evaluator): The evaluator used for validation.
        """
        # Context: Trainer
        # Called after validation run, i.e. at the end of
        # every epoch in the training run.
        # We need to correlate the models in trainer and evaluator
        for _, sm, rt in self._runtime_iterator(evaluator.models):
            rt.train_validation_end(sm)

        self._logic.train_validation_end(evaluator.models)

    def _complete_train_step(
            self, trainer: '_Trainer', outs: Any, block: bool,
            sn: str, sm: torch.nn.Module, rt: 'BaseRuntime',
    ) -> None:
        idx, batch, cback = self.pending_iters[sn][0]
        self.pending_iters[sn] = self.pending_iters[sn][1:]
        # Since async mode is not supported with device splitting
        # we now that there is only ONE submodule, so we
        # can asure that now we can step the optimizers
        self._logic.train_step_optimizers(
            trainer.models, trainer.optimizers, idx)
        if len(self.pending_iters[sn]) == 0:
            del self.pending_iters[sn]
        cback(idx, outs, is_deferred=block)

    def train_step(
            self,
            trainer: '_Trainer',
            batch_idx: int,
            batch: Any,
            complete_fn: Callable[[int, Any], None],
    ) -> None:
        """A training step.

        Args:
            trainer (Trainer): A trainer.
            batch_idx (int): Number of iterations already finished.
            batch (dict of torch.Tensor): Input tensors of this batch.
            complete_fn (callable): A callback function called after
                training step.
        """
        # Batch can be a dict or a tuple of dicts (1 per model in the logic)
        for _, sm, rt in self._runtime_iterator(trainer.models):
            rt.train_pre_step(trainer, sm, batch_idx, batch)

        batch = self._entry_runtime.convert_batch(batch)

        outs = self._logic.train_step(
            trainer.models, trainer.optimizers, batch_idx, batch)

        if self._async:
            # async returns inmediately
            # Check if there is a completed iteration
            # If we enqueue everything first, we will blow up with
            # memory due to the dataloaders being in the background
            # We need to call the tagged runtimes since async mode
            # require different treatment depending on the device.
            for sn, sm, rt in self._runtime_iterator(trainer.models):
                self.pending_iters[sn].append((batch_idx, batch, complete_fn))
                outs = rt.get_pending_result(sm, False)
                if outs is not None:
                    self._complete_train_step(trainer, outs, False, sn, sm, rt)
        else:
            self._logic.train_step_optimizers(
                trainer.models, trainer.optimizers, batch_idx)
            complete_fn(batch_idx, outs)

    def eval_setup(
            self,
            evaluator: '_Evaluator',
            loader: Iterable[Any]
    ) -> None:
        """Called only once when starting a training run.
        When evaluator is not given, this method is not called.

        Args:
            evaluator (Evaluator): The evaluator.
            loader (torch.utils.data.DataLoader): The data loader.
        """
        for _, model in evaluator.models.items():
            model.eval()
        self._setup(evaluator.models, loader)

    def _complete_eval_step(
            self, evaluator: '_Evaluator', outs: Any, block: bool,
            sn: str, sm: torch.nn.Module, rt: 'BaseRuntime',
    ) -> None:
        # This call is deferred
        idx, batch, cback = self.pending_iters[sn][0]
        self.pending_iters[sn] = self.pending_iters[sn][1:]
        if len(self.pending_iters[sn]) == 0:
            del self.pending_iters[sn]
        cback(idx, outs, is_deferred=block)

    def eval_step(
            self,
            evaluator: '_Evaluator',
            batch_idx: int,
            batch: Any,
            complete_fn: Callable[[int, Any], None],
    ) -> None:
        """Evaluation iteration.

        Args:
            evaluator (Evaluator): The evaluator.
            batch_idx (int): Number of iterations already finished.
            batch (dict of torch.Tensor): Input tensors of this batch.
            complete_fn (callable): A callback function called after
                training step.
        """
        for _, sm, rt in self._runtime_iterator(evaluator.models):
            rt.eval_pre_step(evaluator, sm, batch_idx, batch)

        batch = self._entry_runtime.convert_batch(batch)

        outs = self._logic.eval_step(evaluator.models, batch_idx, batch)

        if self._async:
            # Is async returns inmediately
            # Check if there is a completed iteration
            # If we enqueue everything first, we will blow up with
            # memory due to the dataloaders being in the background
            for sn, sm, rt in self._runtime_iterator(evaluator.models):
                self.pending_iters[sn].append((batch_idx, batch, complete_fn))
                outs = rt.get_pending_result(sm, False)
                if outs is not None:
                    self._complete_eval_step(
                        evaluator, outs, False, sn, sm, rt)
        else:
            complete_fn(batch_idx, outs)

    def eval_post_step(
            self,
            evaluator: '_Evaluator',
            batch_idx: int,
            batch: Any,
            outputs: Any,
    ) -> None:
        """A method called after each evaluation step.

        Args:
            evaluator (Evaluator): The evaluator.
            batch_idx (int): Number of iterations already finished.
            batch (dict of torch.Tensor): Input tensors of this batch.
            complete_fn (callable): A callback function called after
                training step.
        """
        # Context: Evaluator
        # Called after eval_step.
        for _, sm, rt in self._runtime_iterator(evaluator.models):
            rt.eval_post_step(evaluator, sm, batch_idx, batch, outputs)
        for out in self._eval_report_keys:
            reporting.report({"val/{}".format(out): outputs[out]})

    def eval_loop_end(self, evaluator: '_Evaluator') -> None:
        """A method called after running all steps of the evaluation.

        Args:
            evaluator (Evaluator): The evaluator.
        """
        if self._async:
            while self.pending_iters:
                for sn, sm, rt in self._runtime_iterator(evaluator.models):
                    outs = rt.get_pending_result(sm, True)
                    if outs is not None:
                        self._complete_eval_step(
                            evaluator, outs, True, sn, sm, rt)

    def train_post_step(
            self,
            trainer: '_Trainer',
            batch_idx: int,
            batch: Any,
            outputs: Any
    ) -> None:
        """A method called after each training step.

        Args:
            trainer (Trainer): The trainer that calls this method.
            batch_idx (int): Number of iterations
            batch (dict of torch.Tensor): Input tensors of this batch.
            outputs (dict of torch.Tensor): Output tensors of this batch.
        """
        # Context: Trainer
        # Called after train_step.
        for _, sm, rt in self._runtime_iterator(trainer.models):
            rt.train_post_step(trainer, sm, batch_idx, batch, outputs)
        for out in self._train_report_keys:
            reporting.report({"train/{}".format(out): outputs[out]})
