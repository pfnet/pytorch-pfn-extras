# mypy: ignore-errors

from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Type, Union, TYPE_CHECKING
)

import torch

import pytorch_pfn_extras.handler as handler_module
from pytorch_pfn_extras.runtime import runtime_registry
from pytorch_pfn_extras.training._transform_model import default_transform_model

if TYPE_CHECKING:
    from pytorch_pfn_extras.runtime._runtime import DeviceLike
    from pytorch_pfn_extras import training
    from pytorch_pfn_extras.training.trigger import TriggerLike
    from pytorch_pfn_extras.training._trainer import _Trainer
    from pytorch_pfn_extras.training._evaluator import _Evaluator
    from pytorch_pfn_extras.training.metrics import MetricType
    from pytorch_pfn_extras import writing


class _Engine:
    def __init__(
            self,
            handler: handler_module.BaseHandler,
            models: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
            **kwargs: Any,
    ) -> None:
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
        self._extensions: List[  # list of (args, kwargs)
            Tuple[Tuple['training.Extension', Optional[str],
                        'TriggerLike', Optional[int]],
                  Dict[str, Any]]] = []
        self._manager_state: Optional[Dict[str, Any]] = None

    def extend(
            self,
            extension: 'training.Extension',
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

    def _setup_manager(self, iters_per_epoch: int) -> None:
        from pytorch_pfn_extras.training import ExtensionsManager
        self._manager = ExtensionsManager(
            self._models, iters_per_epoch=iters_per_epoch, **self._kwargs)
        for ex_args, ex_kwargs in self._extensions:
            self._manager.extend(*ex_args, **ex_kwargs)
        if self._manager_state is not None:
            self.manager.load_state_dict(self._manager_state)

    @property
    def manager(self) -> 'training.ExtensionsManager':
        if self._manager is None:
            raise RuntimeError('the engine is not started yet')
        return self._manager

    @property
    def models(self) -> Dict[str, torch.nn.Module]:
        # TODO(kmaehashi): do we need this convenient interface for handlers?
        return self.manager.raw_models

    @property
    def optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return self.manager.optimizers

    def state_dict(self) -> Dict[str, Any]:
        return self.manager.state_dict()

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        if self._manager is None:
            self._manager_state = to_load
            return
        self.manager.load_state_dict(to_load)

    def run(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError


def create_trainer(
        models: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
        optimizers: Dict[str, torch.optim.Optimizer],
        max_epochs: int,
        *,
        extensions: Optional[List['training.Extension']] = None,
        out_dir: str = 'result',
        stop_trigger: 'TriggerLike' = None,
        writer: Optional['writing.Writer'] = None,
        evaluator: Optional['_Evaluator'] = None,
        device: 'DeviceLike' = 'cpu',
        logic: Optional[handler_module.Logic] = None,
        transform_model: Callable[
            [str, torch.nn.Module], torch.nn.Module] = default_transform_model,
        handler_class: Optional[Type[handler_module.BaseHandler]] = None,
        options: Optional[Dict[str, Any]] = None,
        runtime_options: Optional[Dict[str, Any]] = None,
) -> '_Trainer':
    """Creates a trainer object.

    Args:
        models:
            Map of string to Module or an actual Module.
        optimizers:
            Map of string to Optimizer or an actual Optimizer.
        max_epochs:
            Number of epochs in the whole training loop.
            Ignored if `stop_trigger` is passed as a kwarg.
        extensions:
            List of extensions to be registered to the trainer.
        out_dir:
            Output directory (default: ``result``).
        stop_trigger (trigger, optional):
            Trigger that can be consulted to determine wether training has
            concluded. The default is an interval trigger set to `max_epochs`
        writer:
            Writer that can be used by extensions to write data to custom
            filesystems.
        evaluator:
            Evaluator that is used in evaluation phase.
            If `None` is given, the evaluation is skipped.
            Evaluators can be created with
            :func:`pytorch_pfn_extras.engine.create_evaluator`.
        device (str or torch.device):
            Device name used for selecting a corresponding runtime class.
        logic:
            A logic object. If `None` is given, an logic object is instantiated
            from the default logic class.
        transform_model:
            A function to transform a model structure, often used to unwrap the
            a module from DDP module.
        handler_class:
            A handler class that instantiates a handler object. If `None` is
            given, `ppe.handler.Handler` is used as a default handler class.
        options:
            Options that are set to the handler and logic object.
            See the documentation of `ppe.handler.Handler` and
            `ppe.handler.Logic` for details.
        runtime_options:
            Options that are set to the runtime object. See the documentation
            of `ppe.runtime.PyTorchRuntime` for details.
    """

    options = options.copy() if options else {}
    # TODO(kmaehashi): deprecate specifying 'runtime' key in options
    runtime_options = (
        runtime_options.copy() if runtime_options
        else options.pop('runtime', {}))
    logic = handler_module.Logic() if logic is None else logic
    handler_class = handler_class if handler_class else handler_module.Handler

    entry_runtime_cls = runtime_registry.get_runtime_class_for_device_spec(
        device)
    entry_runtime = entry_runtime_cls(device, runtime_options)
    handler = handler_class(logic, entry_runtime, {})

    # Handler options are popped first, then pass the remainings to Logic.
    handler.consume_options(options)
    logic.consume_options(options)
    if len(options) > 0:
        raise ValueError('Unknown options: ', options)

    from pytorch_pfn_extras.training._trainer import _Trainer
    return _Trainer(
        handler, evaluator=evaluator,
        models=models, optimizers=optimizers, max_epochs=max_epochs,
        extensions=extensions, out_dir=out_dir,
        stop_trigger=stop_trigger, writer=writer,
        transform_model=transform_model,
    )


def create_evaluator(
        models: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
        *,
        progress_bar: bool = False,
        device: 'DeviceLike' = 'cpu',
        metrics: Optional[List['MetricType']] = None,
        logic: Optional[handler_module.Logic] = None,
        handler_class: Optional[Type[handler_module.BaseHandler]] = None,
        options: Optional[Dict[str, Any]] = None,
        runtime_options: Optional[Dict[str, Any]] = None,
) -> '_Evaluator':
    """Creates an evaluator object. The return value of this function is
    expected to be fed to `ppe.engine.create_trainer` as an argument.

    Args:
        models:
            Map of string to :class:`torch.nn.Module` or an actual Module.
            In most cases, this arugment is the same as the `model` arguemnt of
            `ppe.engine.create_trainer`.
        progress_bar:
            If `True`, a progress bar is enabled in evaluation.
        device (str or torch.device):
            Device name used for selecting a corresponding runtime class.
        metrics (list of metrics):
            List of metrics, which computes various quantities and update
            output for the reporting.
        logic:
            A logic object. If `None` is given, an logic object is instantiated
            from the default logic class.
        handler_class:
            A handler class that instantiates a handler object. If `None` is
            given, `ppe.handler.Handler` is used as a default handler class.
        options:
            Options that are set to the handler and logic object.
            See the documentation of `ppe.handler.Handler` and
            `ppe.handler.Logic` for details.
        runtime_options:
            Options that are set to the runtime object. See the documentation
            of `ppe.handler.Handler` for details.
    """

    metrics = metrics if metrics else []
    options = options.copy() if options else {}
    # TODO(kmaehashi): deprecate specifying 'runtime' key in options
    runtime_options = (
        runtime_options.copy() if runtime_options
        else options.pop('runtime', {}))
    logic = handler_module.Logic() if logic is None else logic
    handler_class = handler_class if handler_class else handler_module.Handler

    entry_runtime_cls = runtime_registry.get_runtime_class_for_device_spec(
        device)
    entry_runtime = entry_runtime_cls(device, runtime_options)
    handler = handler_class(logic, entry_runtime, options)

    # Handler options are popped first, then pass the remainings to Logic.
    handler.consume_options(options)
    logic.consume_options(options)
    if len(options) > 0:
        raise ValueError('Unknown options: ', options)

    from pytorch_pfn_extras.training._evaluator import _Evaluator
    return _Evaluator(
        handler,
        models=models,
        progress_bar=progress_bar,
        metrics=metrics,
    )
