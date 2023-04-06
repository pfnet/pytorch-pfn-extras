import contextlib
import types

from typing import (
    Any, Dict, Generator, Iterable, Optional, Set, Tuple, Union, TYPE_CHECKING
)

import torch

from pytorch_pfn_extras.handler._code_block import CodeBlock
from pytorch_pfn_extras.runtime import _autocast

if TYPE_CHECKING:
    from pytorch_pfn_extras.training import Evaluator, Trainer

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
        trainer: 'Trainer',
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
        trainer: 'Trainer',
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

    def train_cleanup(self, module: torch.nn.Module) -> None:
        """A method called only once when compleing a training run.

        Args:
            module (torch.nn.Module): A module.

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
        evaluator: 'Evaluator',
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
        evaluator: 'Evaluator',
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

    def map(
        self,
        func: CodeBlock,
        iterable: Iterable[Any],
        out_keys: Optional[Set[str]] = None,
        device: Any = "cpu",
    ) -> Iterable[Any]:
        """Method called by the user to apply function to iterable efficiently.

        Args:
            func: The function to be executed
            iterable: The data
            out_keys: The output keys that to be moved to the host device
            device: The torch device that contains the final outputs

        Returns:
            The result of `func`
        """
        raise NotImplementedError()

    @classmethod
    @contextlib.contextmanager
    def trace(cls, event_name: Optional[str], arg: Any) -> Generator[None, None, None]:
        """Context manager for tracing PPE events in the custom device tools.

        Args:
            event_name: The name of the event being traced
            arg: Custom argument for the tracer
        """
        yield


class PyTorchRuntime(BaseRuntime):
    """A collections of callback functions for the devices that PyTorch
    supports by default.

    Args:
        device_spec (torch.device or str): The device.
        options (dict, optional): The configuration options.

            * ``'autocast'`` (bool or dict):
                If ``True``, ``torch.cuda.amp.autocast`` is enabled.
                using ``{"enabled": True, "device_type": "cuda"}``
                as autocast options.
                Default is ``False`` which corresponds to the following options
                ``{"enabled": False, "device_type": "cuda"}``
                dict type. If dict, Options to pass to ``torch.autocast``.
                Includes ``device_type``, ``dtype`` among others.
            * ``'grad_scaler'`` (torch.cuda.amp.GradScaler):
                A gradient scaler that outputs are applied to.
    """

    def __init__(
        self, device_spec: DeviceLike, options: Dict[str, Any],
    ) -> None:
        super().__init__(device_spec, options)
        self._grad_scaler = options.get("grad_scaler", None)
        autocast_options = options.get("autocast", False)
        if isinstance(autocast_options, bool):
            autocast_options = {"enabled": autocast_options, "device_type": "cuda"}
        self._autocast = _autocast._AutocastManager(
            autocast_options, self._grad_scaler is not None
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

    def train_cleanup(self, module: torch.nn.Module) -> None:
        pass

    def train_validation_begin(self, module: torch.nn.Module) -> None:
        pass

    def train_validation_end(self, module: torch.nn.Module) -> None:
        pass

    def train_pre_step(
        self,
        trainer: 'Trainer',
        module: torch.nn.Module,
        batch_idx: int,
        batch: Any,
    ) -> None:
        pass

    def train_post_step(
        self,
        trainer: 'Trainer',
        module: torch.nn.Module,
        batch_idx: int,
        batch: Any,
        outs: Any,
    ) -> None:
        pass

    def eval_pre_step(
        self,
        evaluator: 'Evaluator',
        module: torch.nn.Module,
        batch_idx: int,
        batch: Any,
    ) -> None:
        pass

    def eval_post_step(
        self,
        evaluator: 'Evaluator',
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

        for optimizer in code_block.optimizers:
            optimizer.zero_grad()

        # with autocast
        with self._autocast.autocast():
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

        if len(code_block.optimizers) == 0:
            return out

        if self._grad_scaler is not None:
            # TODO support multiple optimizers with grad scaler
            assert len(code_block.optimizers) == 1
            self._grad_scaler.step(code_block.optimizers[0])
            self._grad_scaler.update()
        else:
            for optimizer in code_block.optimizers:
                optimizer.step()

        return out

    def map(
        self,
        func: CodeBlock,
        iterable: Iterable[Any],
        out_keys: Optional[Set[str]] = None,
        device: Any = "cpu",
    ) -> Iterable[Any]:
        for data in iterable:
            # TODO overlap computation and data transfer when using CUDA
            out = func(data)
            if out_keys is not None:
                assert isinstance(out, dict)
                out = {key: out[key] for key in out_keys}
            if isinstance(out, dict):
                out = {k: v.to(device) for k, v in out.items()}
            else:
                out = out.to(device)
            yield out

    @classmethod
    @contextlib.contextmanager
    def trace(cls, event_name: Optional[str], arg: Any) -> Generator[None, None, None]:
        """Context manager for tracing PPE events in the custom device tools.

        Args:
            event_name: The name of the event being traced
            arg: Custom argument for the tracer
        """
        assert event_name is not None
        with torch.autograd.profiler.record_function(event_name):
            yield


def _module_runtime_tag(module: torch.nn.Module) -> Optional[BaseRuntime]:
    return getattr(  # type: ignore[no-any-return]
        module, _RUNTIME_TAG_NAME, None
    )


def _set_module_runtime_tag(
    module: torch.nn.Module, runtime: BaseRuntime
) -> None:
    setattr(module, _RUNTIME_TAG_NAME, runtime)

    def mk_getstate(orig_getstate):  # type: ignore
        def _getstate_without_runtime(self):  # type: ignore
            if orig_getstate is not None:
                state = orig_getstate()
            else:
                state = self.__dict__

            # remove runtime class and getstate
            def _remove_runtime_class(state):  # type: ignore
                state = {k: v for k, v in state.items() if k != _RUNTIME_TAG_NAME}
                for k, v in state.items():
                    if isinstance(v, dict):
                        state[k] = _remove_runtime_class(v)  # type: ignore
                for k in list(state.keys()):
                    if k == "__getstate__":
                        if orig_getstate is not None:
                            state[k] = orig_getstate
                        else:
                            del state[k]
                return state

            return _remove_runtime_class(state)  # type: ignore
        return _getstate_without_runtime

    getstate = None
    if hasattr(module, "__getstate__"):
        getstate = module.__getstate__

    setattr(  # NOQA
        module,
        "__getstate__",
        types.MethodType(mk_getstate(getstate), module)  # type: ignore
    )


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
        # nested runtime tag is ignored
        yield module_name, module
