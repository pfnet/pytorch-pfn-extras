from typing import (
    Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type,
    Union, TYPE_CHECKING,
)

import torch

import pytorch_pfn_extras.handler as handler_module
from pytorch_pfn_extras.runtime import runtime_registry
from pytorch_pfn_extras.training._transform_model import default_transform_model

if TYPE_CHECKING:
    from pytorch_pfn_extras.runtime._runtime import DeviceLike
    from pytorch_pfn_extras.training import extension
    from pytorch_pfn_extras.training.trigger import TriggerLike
    from pytorch_pfn_extras.training._trainer import Trainer
    from pytorch_pfn_extras.training._evaluator import Evaluator
    from pytorch_pfn_extras.training.metrics import MetricType
    from pytorch_pfn_extras import writing


def create_trainer(
        models: Union[torch.nn.Module, Mapping[str, torch.nn.Module]],
        optimizers: Union[torch.optim.Optimizer, Mapping[str, torch.optim.Optimizer]],
        max_epochs: int,
        *,
        extensions: Optional[Sequence[Union['extension.ExtensionLike',
                                            'extension.ExtensionEntry']]] = None,
        out_dir: str = 'result',
        stop_trigger: 'TriggerLike' = None,
        writer: Optional['writing.Writer'] = None,
        evaluator: Optional[Union[
            'Evaluator', Tuple['Evaluator', 'TriggerLike'],
            Mapping[str, Union['Evaluator', Tuple['Evaluator', 'TriggerLike']]]
        ]] = None,
        device: 'DeviceLike' = 'cpu',
        logic: Optional[handler_module.BaseLogic] = None,
        transform_model: Callable[
            [str, torch.nn.Module], torch.nn.Module] = default_transform_model,
        handler_class: Optional[Type[handler_module.Handler]] = None,
        options: Optional[Dict[str, Any]] = None,
        runtime_options: Optional[Mapping[str, Any]] = None,
        profile: Optional[torch.profiler.profile] = None,  # type: ignore[name-defined]
        **kwargs: Any,
) -> 'Trainer':
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
        profile:
            A `torch.profiler.profile` object to collect the performance
            metrics.
    """

    options = options.copy() if options else {}
    # TODO(kmaehashi): deprecate specifying 'runtime' key in options
    runtime_options = dict(
        runtime_options if runtime_options
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

    from pytorch_pfn_extras.training._trainer import Trainer
    return Trainer(
        handler, evaluator=evaluator,
        models=models, optimizers=optimizers, max_epochs=max_epochs,
        extensions=extensions, out_dir=out_dir,
        stop_trigger=stop_trigger, writer=writer,
        transform_model=transform_model,
        profile=profile,
        **kwargs,
    )


def create_evaluator(
        models: Union[torch.nn.Module, Mapping[str, torch.nn.Module]],
        *,
        progress_bar: bool = False,
        device: 'DeviceLike' = 'cpu',
        metrics: Optional[Sequence['MetricType']] = None,
        logic: Optional[handler_module.Logic] = None,
        handler_class: Optional[Type[handler_module.Handler]] = None,
        options: Optional[Dict[str, Any]] = None,
        runtime_options: Optional[Mapping[str, Any]] = None,
        profile: Optional[torch.profiler.profile] = None,  # type: ignore[name-defined]
) -> 'Evaluator':
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
        profile:
            A `torch.profiler.profile` object to collect the performance
            metrics.
    """

    metrics = metrics if metrics else []
    options = options.copy() if options else {}
    # TODO(kmaehashi): deprecate specifying 'runtime' key in options
    runtime_options = dict(
        runtime_options if runtime_options
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

    from pytorch_pfn_extras.training._evaluator import Evaluator
    return Evaluator(
        handler,
        models=models,
        progress_bar=progress_bar,
        metrics=metrics,
        profile=profile,
    )
