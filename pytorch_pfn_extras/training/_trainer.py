import collections.abc
import contextlib
import queue
import time
import warnings
from typing import (
    Any, Dict, Generator, Iterable, List, Mapping, Optional, Tuple, Union, TYPE_CHECKING
)

import torch

from pytorch_pfn_extras import training
from pytorch_pfn_extras.training import extension as extension
from pytorch_pfn_extras.training import trigger as trigger_module
import pytorch_pfn_extras.reporting as reporting
from pytorch_pfn_extras.profiler import record

from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol
from pytorch_pfn_extras.training.trigger import Trigger, TriggerLike

if TYPE_CHECKING:
    from pytorch_pfn_extras import handler as handler_module
    from pytorch_pfn_extras.training._evaluator import Evaluator
    from pytorch_pfn_extras.profiler._time_summary import _ReportNotification


@contextlib.contextmanager
def _nullcontext() -> Generator[None, None, None]:
    # contextlib.nullcontext equivalent, needed for Python 3.6 support.
    yield


class Trainer:
    def __init__(
            self,
            handler: 'handler_module.BaseHandler',
            *,
            evaluator: Optional[Union[
                'Evaluator', Tuple['Evaluator', TriggerLike],
                Mapping[str, Union['Evaluator', Tuple['Evaluator', TriggerLike]]]]],
            models: Union[torch.nn.Module, Mapping[str, torch.nn.Module]],
            profile: Optional[torch.profiler.profile] = None,  # type: ignore[name-defined]
            **kwargs: Any,
    ):
        self.handler = handler
        self._manager: Optional['training.ExtensionsManager'] = None

        # The followings are used when setting up a manager instance
        if not isinstance(models, dict):
            if not isinstance(models, torch.nn.Module):
                raise ValueError(
                    'model must be an instance of dict or toch.nn.Module')
            self._models = {'main': models}
        else:
            self._models = models
        self._kwargs = kwargs
        self._profile = profile
        self._enable_profile = kwargs.get('enable_profile', profile is not None)
        self._extensions: List[  # list of (args, kwargs)
            Tuple[Tuple[
                Union['extension.ExtensionLike', extension.ExtensionEntry],
                Optional[str], 'TriggerLike', Optional[int]
            ], Dict[str, Any]]] = []
        self._manager_state: Optional[Dict[str, Any]] = None

        self._evaluators: Dict[str, Tuple['Evaluator', TriggerLike]] = {}
        if evaluator is None:
            evaluator = {}
        elif not isinstance(evaluator, collections.abc.Mapping):
            evaluator = {"Evaluator": evaluator}
        if isinstance(evaluator, collections.abc.Mapping):
            for n, e in evaluator.items():
                self._evaluators[n] = e if isinstance(e, tuple) else (e, (1, 'epoch'))
        self.val_loader = None

    def extend(
            self,
            extension: Union['extension.ExtensionLike', extension.ExtensionEntry],
            name: Optional[str] = None,
            trigger: 'TriggerLike' = None,
            priority: Optional[int] = None,
            *,
            call_before_training: bool = False,
            **kwargs: Any,
    ) -> None:
        if self._manager is not None:
            raise RuntimeError('cannot extend after starting the engine')
        self._extensions.append(
            ((extension, name, trigger, priority),
             dict(call_before_training=call_before_training, **kwargs)))

    def _setup_manager(self, iters_per_epoch: int) -> 'training.ExtensionsManager':
        from pytorch_pfn_extras.training import ExtensionsManager
        self._manager = ExtensionsManager(
            self._models, iters_per_epoch=iters_per_epoch, **self._kwargs)
        for ex_args, ex_kwargs in self._extensions:
            self._manager.extend(*ex_args, **ex_kwargs)
        if self._manager_state is not None:
            self.manager.load_state_dict(self._manager_state)
        return self._manager

    @property
    def manager(self) -> 'training.ExtensionsManager':
        if self._manager is None:
            raise RuntimeError('the engine is not started yet')
        return self._manager

    @property
    def models(self) -> Mapping[str, torch.nn.Module]:
        # TODO(kmaehashi): do we need this convenient interface for handlers?
        return self.manager.raw_models

    @property
    def optimizers(self) -> Mapping[str, torch.optim.Optimizer]:
        return self.manager.optimizers

    def state_dict(self) -> Dict[str, Any]:
        return self.manager.state_dict()

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        if self._manager is None:
            self._manager_state = to_load
            return
        self.manager.load_state_dict(to_load)

    @property
    def epoch(self) -> int:
        return self.manager.epoch

    @property
    def epoch_detail(self) -> float:
        return self.manager.epoch_detail

    @property
    def iteration(self) -> int:
        return self.manager.iteration

    @property
    def is_before_training(self) -> bool:
        return self.manager.iteration == 0

    @property
    def stop_trigger(self) -> Trigger:
        return self._stop_trigger

    @stop_trigger.setter
    def stop_trigger(self, trigger: Trigger) -> None:
        self._stop_trigger = trigger

    @property
    def evaluator(self) -> Optional['Evaluator']:
        if len(self._evaluators) == 0:
            return None
        if len(self._evaluators) == 1:
            return next(iter(self._evaluators.values()))[0]
        raise ValueError('multiple evaluators are registered.')

    def get_optimizer(self, name: str) -> torch.optim.Optimizer:
        return self.manager.optimizers[name]

    def set_optimizer(self, name: str, optimizer: torch.optim.Optimizer) -> None:
        self.manager.optimizers[name] = optimizer  # type: ignore[index]

    def is_epoch_last_iter(self, idx: int) -> bool:
        return (idx + 1) == (self.manager._iters_per_epoch)

    def _complete_step(
            self,
            idx: int,
            outs: Any,
    ) -> None:
        c_idx = self._idxs.get()
        # Asure that iterations complete in order
        if c_idx != idx:
            raise RuntimeError(
                'Completed a not expected iteration. '
                '{} was expected but completion of {} happened'.format(
                    c_idx, idx)
            )
        x = self._inputs.get()
        begin = self._times.get()
        (
            record_iteration,
            record_run_iteration,
            record_train_step,
        ) = self._profile_records.get()
        self.handler.train_post_step(self, idx, x, outs)
        reporting.report({"elapsed_time": time.time() - begin})

    def run(self,
            train_loader: Iterable[Any],
            val_loader: Optional[Iterable[Any]] = None,
            *,
            train_len: Optional[int] = None,
            eval_len: Optional[int] = None) -> None:
        """Executes the training loop.

        Args:
            train_loader (torch.utils.data.DataLoader):
                A data loader for training.
            val_loader (torch.utils.data.DataLoader, optional):
                A data loader passed to ``Evaluator.run()``.
            train_len (int, optional):
                The number of iterations per one training epoch. The default
                value is inferred from the size of training data loader.
            eval_len (int, optional):
                The number of iterations per one evaluation epoch, passed
                to ``Evaluator.run()``

        .. seealso::
            - :meth:`pytorch_pfn_extras.training._evaluator.Evaluator`
        """
        if train_len is None:
            train_len = len(train_loader)  # type: ignore[arg-type]
        if eval_len is None and val_loader is not None:
            eval_len = len(val_loader)  # type: ignore[arg-type]

        self._train_len = train_len
        self._eval_len = eval_len

        device = self.handler._entry_runtime.device_spec  # type: ignore[attr-defined]

        class _EvaluatorExt:
            def __init__(
                    self,
                    trainer: 'Trainer',
                    evaluator: 'Evaluator',
                    val_loader: Optional[Iterable[Any]],
                    eval_len: Optional[int],
            ) -> None:
                self.needs_model_state = True
                self._trainer = trainer
                self._evaluator = evaluator
                self._val_loader = val_loader
                self._eval_len = eval_len

            def __call__(self, manager: ExtensionsManagerProtocol) -> None:
                evaluator = self._evaluator
                if self._val_loader is None:
                    raise ValueError('"val_loader" is not given.')
                evaluator.handler.train_validation_begin(self._trainer, evaluator)
                evaluator.run(self._val_loader, eval_len=self._eval_len)
                evaluator.handler.train_validation_end(self._trainer, evaluator)

        if self._manager is None:
            self._manager = self._setup_manager(train_len)
            for name, (evaluator, trigger) in self._evaluators.items():
                # Register the evaluator as an extension to the manager
                # To be triggered with the correct timing
                self._manager.extend(
                    _EvaluatorExt(self, evaluator, val_loader, eval_len),
                    name=name,
                    trigger=trigger_module.get_trigger(trigger),
                    priority=extension.PRIORITY_WRITER,
                )
            self.handler.train_setup(self, train_loader)
            if len(self._evaluators) == 0:
                if val_loader is not None:
                    warnings.warn(
                        '`val_loader` is given whereas the evaluator is missing.',
                        UserWarning)
            else:
                if val_loader is None:
                    raise ValueError('`val_loader` is required')
                for _, (evaluator, _) in self._evaluators.items():
                    evaluator.handler.eval_setup(evaluator, val_loader)

        with self._profile or _nullcontext() as prof:
            while not self.manager.stop_trigger:
                self.handler.train_epoch_begin(self, train_loader)

                # When iterations are completed in the callback
                # This is needed to avoid being constantly passing parameters
                self._idxs: 'queue.Queue[int]' = queue.Queue()
                self._inputs: 'queue.Queue[Any]' = queue.Queue()
                self._times: 'queue.Queue[float]' = queue.Queue()
                self._observed: 'queue.Queue[reporting.Observation]' = queue.Queue()
                # Iterator must be created after `train_epoch_begin` as it may be
                #  using a DistributedSampler.
                loader_iter = iter(train_loader)
                self._profile_records: 'queue.Queue[List[_ReportNotification]]' \
                    = queue.Queue()
                for idx in range(train_len):
                    with record(
                        "pytorch_pfn_extras.training.Trainer:iteration",
                        use_cuda=torch.cuda.is_available(),
                        enable=self._enable_profile,
                        device=device
                    ) as ntf0:
                        try:
                            with record(
                                "pytorch_pfn_extras.training.Trainer:get_data",
                                enable=self._enable_profile,
                                device=device
                            ):
                                x = next(loader_iter)
                        except StopIteration:
                            loader_iter = iter(train_loader)
                            with record(
                                "pytorch_pfn_extras.training.Trainer:get_data",
                                enable=self._enable_profile,
                                device=device
                            ):
                                x = next(loader_iter)
                        begin = time.time()
                        self._idxs.put(idx)
                        self._inputs.put(x)
                        self._times.put(begin)
                        try:
                            with record(
                                "pytorch_pfn_extras.training.Trainer:run_iteration",
                                use_cuda=torch.cuda.is_available(),
                                enable=self._enable_profile,
                                device=device
                            ) as ntf1, \
                                    self.manager.run_iteration():
                                self._observed.put(self.manager.observation)
                                with record(
                                    "pytorch_pfn_extras.training.Trainer:train_step",
                                    use_cuda=torch.cuda.is_available(),
                                    enable=self._enable_profile,
                                    device=device
                                ) as ntf2:
                                    self._profile_records.put([ntf0, ntf1, ntf2])
                                    self.handler.train_step(
                                        self, idx, x, complete_fn=self._complete_step)
                                    # Check if the callback was called
                        except Exception:
                            # The manager has errored and called the extensions
                            # on_error. However the manager is reusable
                            # so training can continue and extensions state is not
                            # finalized. On the other hand, the trainer is not
                            # reusable, so we finalize the extensions here.
                            self.manager.finalize()
                            raise

                    if prof is not None:
                        prof.step()  # type: ignore[no-untyped-call]
                    # In some cases, DataLoaders are continuos
                    # And will keep yielding results even if the epoch
                    # is completed. We forcefully exit at the end of
                    # every epoch
                    if self.is_epoch_last_iter(idx) or self.manager.stop_trigger:
                        break
                # In handlers that support a completely Async model train_epoch_end
                # Will take care of completing pending work
                self.handler.train_epoch_end(self)
            if prof is not None:
                prof.on_trace_ready = None
        self.handler.train_cleanup(self)


# For backward compatibility
_Trainer = Trainer
