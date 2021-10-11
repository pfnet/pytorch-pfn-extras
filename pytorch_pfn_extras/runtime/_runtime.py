# mypy: ignore-errors

from typing import (
    Any, Dict, Generator, Iterable, Optional, Tuple, Union
)

import torch

from pytorch_pfn_extras.training import Evaluator, Trainer

_RUNTIME_TAG_NAME = '_ppe_runtime'

DeviceLike = Union[str, torch.device]


class BaseRuntime:
    """A base class for collections of device-specific callback functions.

    The function attributes of this class will be called from
    ``ppe.to`` or ``ppe.handler.Handler``.

    ``ppe.runtime.runtime_registry`` stores the runtime classes and
    dispatches them by feeding the corresponding name string as an input.

    Args:
        device_spec (torch.device or str):
            The device that modules and tensors are transferred to.
        options (dict):
            A configuration dictionary that can be used from runtime method.
    """

    def __init__(
            self,
            device_spec: DeviceLike,
            options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device_spec = device_spec
        self.options = options

    def convert_batch(self, args: Any) -> Any:
        """Transfers the given batch to the specific device.

        Args:
            args (object): A batch data of any type.

        Returns:
            A batch data transferred to the specific device
            of the same type as input.
        """

        # this should be called with the runtime associated to a model
        # or a model part
        if isinstance(args, tuple) and hasattr(args, '_fields'):
            return args.__class__(
                **{k: self.move_tensor(getattr(args, k)) for k in args._fields})
        if isinstance(args, dict):
            return {
                k: self.move_tensor(v) if isinstance(v, torch.Tensor) else v
                for k, v in args.items()
            }
        if isinstance(args, (list, tuple)):
            return [
                self.move_tensor(v) if isinstance(v, torch.Tensor) else v
                for v in args
            ]
        if isinstance(args, torch.Tensor):
            return self.move_tensor(args)
        return args

    def move_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """Transfers the module to the specific device.

        Before this method is called, ``ppe.to`` will add this class as
        an new attribute ("_ppe_runtime") to the input module.

        Args:
            module (torch.nn.Module): A module.

        Returns:
            A module transferred to the specific device.
        """
        raise NotImplementedError()

    def move_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transfers the tensor to the specific device.

        Args:
            tensor (torch.Tensor): A tensor.

        Returns:
            A tensor transferred to the specific device.
        """
        raise NotImplementedError()

    def initialize_module(
            self,
            module: torch.nn.Module,
            loader_or_batch: Optional[Union[Iterable[Any], torch.Tensor]],
            optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Initializes the module at the beginning of training or inference.

        Args:
            module (torch.nn.Module):
                A module.
            loader_or_batch (DataLoader or torch.Tensor):
                A data loader or a tensor.
            optimizer (Optimizer or None):
                An optimizer. This argument is sometimes used to copy
                LR from the original optimizer to the training model.

        Returns: None
        """
        raise NotImplementedError()

    def train_epoch_begin(self, module: torch.nn.Module) -> None:
        """Preprocess of each epoch.

        Args:
            module (torch.nn.Module): A module.

        Returns: None
        """
        raise NotImplementedError()

    def train_pre_step(
            self,
            trainer: Trainer,
            module: torch.nn.Module,
            batch_idx: int,
            batch: Any,
    ) -> None:
        """Preprocess of each step.

        This method is called at the beginning of every steps: the set of
        (typically one) iterations and an update.

        Args:
            trainer (Trainer): A trainer.
            module (torch.nn.Module): A module.
            batch_idx (int): The batch index.
            batch (list of torch.Tensor):
                The list of input tensors of this batch.

        Returns: None
        """
        raise NotImplementedError()

    def train_post_step(
            self,
            trainer: Trainer,
            module: torch.nn.Module,
            batch_idx: int,
            batch: Any,
            outs: Any,
    ) -> None:
        """Postprocess of each step.

        This method is called at the end of every steps: the set of
        (typically one) iterations and an update.

        Args:
            trainer (Trainer): A trainer.
            module (torch.nn.Module): A module.
            batch_idx (int): The batch index.
            batch (list of torch.Tensor):
                 The list of input tensors of this batch.
            outs: (list of torch.Tensor):
                 The list of output tensors of this batch.

        Returns: None
        """
        raise NotImplementedError()

    def train_validation_begin(self, module: torch.nn.Module) -> None:
        """The method called before each evaluation.

        Args:
            module (torch.nn.Module): A module.

        Returns: None
        """
        raise NotImplementedError()

    def train_validation_end(self, module: torch.nn.Module) -> None:
        """The method called after each evaluation.

        Args:
            module (torch.nn.Module): A module.

        Returns: None
        """
        raise NotImplementedError()

    def eval_pre_step(
            self,
            evaluator: Evaluator,
            module: torch.nn.Module,
            batch_idx: int,
            batch: Any,
    ) -> None:
        """The method called at the beginning of each evaluation.

        Args:
            evaluator (Evaluator): An evaluator.
            module (torch.nn.Module): A module.
            batch_idx (int): The batch index.
            batch (list of torch.Tensor):
                 The list of input tensors of this batch.

        Returns: None
        """
        raise NotImplementedError()

    def eval_post_step(
            self,
            evaluator: Evaluator,
            module: torch.nn.Module,
            batch_idx: int,
            batch: Any,
            outs: Any,
    ) -> None:
        """The method called at the end of each evaluation.

        Args:
            evaluator (Evaluator): An evaluator.
            module (torch.nn.Module): A module.
            batch_idx (int): The batch index.
            batch (list of torch.Tensor):
                 The list of input tensors of this batch.
            outs: (list of torch.Tensor):
                 The list of output tensors of this batch.

        Returns: None
        """
        raise NotImplementedError()


class PyTorchRuntime(BaseRuntime):
    """A collections of callback functions for the devices that PyTorch
    supports by default.

    Args:
        device_spec (torch.device or str): The device.
    """

    def move_module(self, module: torch.nn.Module) -> torch.nn.Module:
        return module.to(self.device_spec)

    def move_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device_spec)

    def initialize_module(
            self,
            module: torch.nn.Module,
            loader_or_batch: Optional[Union[Iterable[Any], torch.Tensor]],
            optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        pass

    def train_epoch_begin(self, module: torch.nn.Module) -> None:
        pass

    def train_validation_begin(self, module: torch.nn.Module) -> None:
        pass

    def train_validation_end(self, module: torch.nn.Module) -> None:
        pass

    def train_pre_step(
            self,
            trainer: Trainer,
            module: torch.nn.Module,
            batch_idx: int,
            batch: Any,
    ) -> None:
        pass

    def train_post_step(
            self,
            trainer: Trainer,
            module: torch.nn.Module,
            batch_idx: int,
            batch: Any,
            outs: Any,
    ) -> None:
        pass

    def eval_pre_step(
            self,
            evaluator: Evaluator,
            module: torch.nn.Module,
            batch_idx: int,
            batch: Any,
    ) -> None:
        pass

    def eval_post_step(
            self,
            evaluator: Evaluator,
            module: torch.nn.Module,
            batch_idx: int,
            batch: Any,
            outs: Any,
    ) -> None:
        pass


def _module_runtime_tag(module: torch.nn.Module) -> BaseRuntime:
    return getattr(  # type: ignore[no-any-return]
        module, _RUNTIME_TAG_NAME, None)


def _set_module_runtime_tag(
        module: torch.nn.Module, runtime: BaseRuntime) -> None:
    return setattr(module, _RUNTIME_TAG_NAME, runtime)


def named_runtime_modules(
        module: torch.nn.Module,
        module_name: str = '',
        first_level: bool = True,
        recursive: bool = True,
) -> Generator[Tuple[str, torch.nn.Module], Tuple[str, torch.nn.Module], None]:
    # This can be invoked with no containarized modules
    # to look for submodules that hold containers
    if _module_runtime_tag(module) is None:
        if first_level or recursive:
            for name, sm in module.named_children():
                yield from named_runtime_modules(sm, name, False, recursive)
    else:
        yield module_name, module
