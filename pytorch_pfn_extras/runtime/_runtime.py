import contextlib

from typing import Any, Dict, Generator, Iterable, Optional, Tuple, Union

import torch

from pytorch_pfn_extras.handler._code_block import CodeBlock
from pytorch_pfn_extras.training import Evaluator, Trainer

_amp_enabled = False


try:
    import torch.cuda.amp

    _amp_enabled = torch.cuda.is_available() and hasattr(
        torch.cuda.amp, "autocast"
    )
except ImportError:
    pass


@contextlib.contextmanager
def _autocast(enabled: bool = True) -> Generator[None, None, None]:
    if _amp_enabled:
        with torch.cuda.amp.autocast(enabled):  # type: ignore[no-untyped-call]
            yield
    else:
        yield


_RUNTIME_TAG_NAME = "_ppe_runtime"

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
        self, device_spec: DeviceLike, options: Dict[str, Any],
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
        if isinstance(args, tuple) and hasattr(args, "_fields"):
            # namedtuple
            return args._replace(  # type: ignore[attr-defined]
                **self._convert_batch_dict(args._asdict()))  # type: ignore
        if isinstance(args, dict):
            return self._convert_batch_dict(args)
        if isinstance(args, (list, tuple)):
            return [
                self.move_tensor(v) if isinstance(v, torch.Tensor) else v
                for v in args
            ]
        if isinstance(args, torch.Tensor):
            return self.move_tensor(args)
        return args

    def _convert_batch_dict(self, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: self.move_tensor(v) if isinstance(v, torch.Tensor) else v
            for k, v in args.items()
        }

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

    def train_epoch_end(self, module: torch.nn.Module) -> None:
        """Completion of each epoch.

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

    def execute(self, code_block: CodeBlock, batch: Any,) -> Any:
        """Method called by the CodeBlocks API to do device dependent execution.

        Args:
            code_block (CodeBlock): The codeblock requesting execution.
            batch (dict of str, torch.Tensor):
                The input tensors of this batch.

        Returns:
            The results of executing the codeblock on this runtime.
        """
        raise NotImplementedError()


class PyTorchRuntime(BaseRuntime):
    """A collections of callback functions for the devices that PyTorch
    supports by default.

    Args:
        device_spec (torch.device or str): The device.
        options (dict, optional): The configuration options.

            * ``'autocast'`` (bool):
                If ``True``, ``torch.cuda.amp.autocast`` is enabled.
                Default is ``False``.
            * ``'grad_scaler'`` (torch.cuda.amp.GradScaler):
                A gradient scaler that outputs are applied to.
    """

    def __init__(
        self, device_spec: DeviceLike, options: Dict[str, Any],
    ) -> None:
        super().__init__(device_spec, options)
        self._grad_scaler = options.get("grad_scaler", None)
        self._autocast = options.get("autocast", False)
        if not _amp_enabled:
            if self._grad_scaler is not None or self._autocast:
                raise RuntimeError(
                    "Requested AMP features but torch.cuda.amp"
                    " is not enabled"
                )

        if self._grad_scaler is not None:
            if not isinstance(self._grad_scaler, torch.cuda.amp.GradScaler):
                raise RuntimeError(
                    "grad_scaler should be a "
                    "torch.cuda.amp.GradScaler object"
                )

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

    def train_epoch_end(self, module: torch.nn.Module) -> None:
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

    def execute(self, code_block: CodeBlock, batch: Any,) -> Any:
        # Run forward, backward and optimize steps depending on codeblock opts
        if self._grad_scaler is None:

            def _scale(x: torch.Tensor) -> torch.Tensor:
                return x

        else:

            def _scale(x: torch.Tensor) -> torch.Tensor:
                return self._grad_scaler.scale(x)  # type: ignore[no-any-return]

        if code_block.optimizer is not None:
            code_block.optimizer.zero_grad()

        # with autocast
        with _autocast(enabled=self._autocast):
            out = code_block.func(**batch)

        # codeblocks return Dicts-per-se so it is not necessary to normalize
        if code_block.backprop:
            if code_block.backprop_from is None:
                for v in out.values():
                    if (
                        isinstance(v, torch.Tensor)
                        and v.grad_fn is not None
                        and v.numel() == 1
                        and (
                            v.dtype.is_floating_point
                            or v.dtype.is_complex
                        )
                    ):
                        _scale(v).backward()  # type: ignore[no-untyped-call]
            else:
                _scale(out[code_block.backprop_from]).backward()  # type: ignore

        if code_block.optimizer is None:
            return out

        if self._grad_scaler is not None:
            self._grad_scaler.step(code_block.optimizer)
            self._grad_scaler.update()
        else:
            code_block.optimizer.step()

        return out


def _module_runtime_tag(module: torch.nn.Module) -> Optional[BaseRuntime]:
    return getattr(  # type: ignore[no-any-return]
        module, _RUNTIME_TAG_NAME, None
    )


def _set_module_runtime_tag(
    module: torch.nn.Module, runtime: BaseRuntime
) -> None:
    return setattr(module, _RUNTIME_TAG_NAME, runtime)


def named_runtime_modules(
    module: torch.nn.Module,
    module_name: str = "",
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
        if first_level or recursive:
            for sm in module.children():
                for descendant in sm.modules():
                    if _module_runtime_tag(descendant) is not None:
                        raise ValueError("Runtimes cannot be nested.")
        yield module_name, module
