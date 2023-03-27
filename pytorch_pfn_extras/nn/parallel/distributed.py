import logging

from contextlib import contextmanager
from collections import OrderedDict
from typing import (
    Any, Callable, Dict, Generator, List, Mapping, Optional, Sequence, Tuple,
    TypeVar, Union
)

import torch
from torch import nn
from torch import distributed as dist
from torch.utils import hooks
from torch.autograd import Variable
from torch.autograd.profiler import record_function
import threading

from pytorch_pfn_extras.profiler import record

logger = logging.getLogger(__name__)

Tensors = Union[Tuple[torch.Tensor, ...], torch.Tensor]
DistFunc = Callable[[Sequence[torch.Tensor], Optional[dist.ProcessGroup]], None]
HookFun = Callable[['DistributedDataParallel'], None]


class _ForEachWrapper:
    """A wrapper class of `torch._foreach_xxx` function.

    There are two options:
    - torch with python for loop
    - torch._foreach_xxx
    """
    def __init__(self) -> None:
        self.flatten = torch._utils._flatten_dense_tensors
        self.unflatten = torch._utils._unflatten_dense_tensors

        self._enable_foreach = (hasattr(torch, "_foreach_add")
                                and hasattr(torch, '_foreach_zero_'))
        if not self._enable_foreach:
            logger.warning(
                "torch does not have _foreach_xxx functions."
                " Please use newer torch")

    def multi_tensor_scale(
            self,
            src: Sequence[torch.Tensor],
            dst: Sequence[torch.Tensor],
            scale: float,
    ) -> None:
        with torch.no_grad():  # type: ignore[no-untyped-call]
            # _foreach_zero for long type is not supported in CUDA
            if self._enable_foreach and src[0].is_floating_point():
                # scale
                val = torch._foreach_mul(tuple(src), scale)
                # copy tensor
                torch._foreach_zero_(tuple(dst))
                torch._foreach_add_(tuple(dst), val)
            else:
                for s, d in zip(src, dst):
                    d.copy_(s * scale)


foreach_wrapper: Optional[_ForEachWrapper] = None
_apex_wrapper_mutex = threading.Lock()


def get_foreach_wrapper() -> _ForEachWrapper:
    global foreach_wrapper
    if foreach_wrapper is None:
        with _apex_wrapper_mutex:
            if foreach_wrapper is None:
                foreach_wrapper = _ForEachWrapper()
    return foreach_wrapper


def _reduce(
        values: Sequence[torch.Tensor],
        group: Optional[dist.ProcessGroup],
) -> None:
    size = sum([v.numel() for v in values])

    # flatten values to improve the runtime perfomance of all-reduce
    coalesced = torch.empty(size, device=values[0].device,
                            dtype=values[0].dtype)
    coalesced_views = get_foreach_wrapper().unflatten(  # type: ignore[no-untyped-call]
        coalesced, values)
    get_foreach_wrapper().multi_tensor_scale(values, coalesced_views, 1.0)

    with record(
        "torch.distributed.all_reduce", use_cuda=torch.cuda.is_available()
    ):
        dist.all_reduce(coalesced, group=group)  # type: ignore[no-untyped-call]

    # unflatten values
    get_foreach_wrapper().multi_tensor_scale(
        coalesced_views, values,
        1.0 / dist.get_world_size(group)  # type: ignore[no-untyped-call]
    )


def _broadcast(
        values: Sequence[torch.Tensor],
        group: Optional[dist.ProcessGroup]
) -> None:
    with torch.no_grad():  # type: ignore[no-untyped-call]
        coalesced = get_foreach_wrapper().flatten(  # type: ignore[no-untyped-call]
            values)
        with record(
            "torch.distributed.broadcast", use_cuda=torch.cuda.is_available()
        ):
            dist.broadcast(coalesced, 0, group=group)  # type: ignore[no-untyped-call]
        src = get_foreach_wrapper().unflatten(  # type: ignore[no-untyped-call]
            coalesced, values)
        get_foreach_wrapper().multi_tensor_scale(src, values, 1.0)


def _group_by_type(values: Sequence[Optional[torch.Tensor]]) -> List[List[torch.Tensor]]:
    groups: Dict[torch.dtype, List[torch.Tensor]] = {}
    for value in values:
        if value is None:
            continue
        if value.dtype not in groups:
            groups[value.dtype] = []
        groups[value.dtype].append(value)
    out = list(groups.items())
    out.sort(key=lambda x: str(x[0]))
    return [value for _, value in out]


class DistributedDataParallel(nn.Module):
    """Module for distributed data parallelism

    This class synchronizes the gradients and the buffers after
    backward computations.

    Args:
        module: torch.nn.Module object to be trained
        broadcast_buffers: Boolean flag to broadcast buffers after backward
            computations. Broadcasting buffers may be helpful when the module
            includes BatchNormalization.
            However, it will degrade training throughput.
            (default: `True`)
        negotiate_grads: Boolean flag to choose gradients to be sent before
            all-reduce. This flag is necessary when the computation graph of
            the module is dynamic.
            (default: `True`)
        process_group: Process group used for broadcasting and reducing.
            (default: `torch.distributed.group.WORLD`)
        reduce_function: All-reduce function
        broadcast_function: Broadcast function
    """

    _unused_parameters = ["device_ids", "output_device", "dim",
                          "find_unused_parameters", "check_reduction",
                          "gradient_as_bucket_view"]

    def __init__(
            self,
            module: nn.Module,
            broadcast_buffers: bool = True,
            negotiate_grads: bool = True,
            process_group: Optional[dist.ProcessGroup] = None,
            reduce_function: Optional[DistFunc] = None,
            broadcast_function: Optional[DistFunc] = None,
            **kwargs: Any
    ) -> None:
        """
        This module receives keyword arguments for the compatibility with
        `torch.nn.parallel.DistributedDataParallel`.
        It shows a warning when setting the ignored arguments.
        """
        super().__init__()  # type: ignore[no-untyped-call]

        for name in DistributedDataParallel._unused_parameters:
            if name in kwargs:
                logger.warning(
                    "ppe.nn.parallel.DistributedDataParallel"
                    " ignores {}".format(name)
                )

        if process_group is None:
            process_group = dist.group.WORLD

        self.module = module
        self._broadcast_buffers = broadcast_buffers
        self._negotiate_grads = negotiate_grads
        self._process_group = process_group
        self._reduce_function = reduce_function or _reduce
        self._broadcast_function = broadcast_function or _broadcast

        self._device = list(self.parameters())[0].device

        self._sorted_param_keys = [key for key, _ in self.named_parameters()]
        self._sorted_param_keys.sort()
        self._sorted_buffer_keys = [key for key, _ in self.named_buffers()]
        self._sorted_buffer_keys.sort()

        self._comm_hooks: Dict[int, HookFun] = OrderedDict()

        self._require_sync = True
        self._show_send_gpu_warning = False

        # synchronize initial parameters and buffers
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        values = \
            [buffers[name] for name in self._sorted_buffer_keys] + \
            [params[name] for name in self._sorted_param_keys]
        if dist.is_initialized():  # type: ignore[no-untyped-call]
            groups = _group_by_type(values)
            for group in groups:
                _broadcast(group, self._process_group)
        else:
            logger.warning("torch.distributed is not initialized")

        # add hook to launch synchronization
        self.register_backward_hook(self._backward_hook)

    @contextmanager
    def no_sync(self) -> Generator[None, None, None]:
        """A context manager to disable synchronization after backward
        """
        prev = self._require_sync
        self._require_sync = False
        try:
            yield
        finally:
            self._require_sync = prev

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        args = self._input_to_device(args)
        kwargs = self._input_to_device(kwargs)
        return self.module(*args, **kwargs)

    def load_state_dict(
            self,
            state_dict: 'Mapping[str, torch.Tensor]',
            strict: bool = True,
    ) -> None:
        self.module.load_state_dict(state_dict, strict=strict)  # type: ignore[arg-type]

    T_destination = TypeVar('T_destination', bound=Mapping[str, torch.Tensor])

    def state_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        return self.module.state_dict()

    def register_comm_hook(self, hook: HookFun) -> hooks.RemovableHandle:
        """Registers a hook function. This module will invoke the hook before
        starting the synchronization.

        Args:
        hook: Callable object that will be invoked before synchronization
        """
        handle = hooks.RemovableHandle(self._comm_hooks)
        self._comm_hooks[handle.id] = hook
        return handle

    def _backward_hook(
            self,
            module: torch.nn.Module,
            gin: Tensors,
            gout: Tensors
    ) -> None:
        def _synchronize() -> None:
            if not self._require_sync:
                return

            for hook in self._comm_hooks.values():
                hook(self)

            with record_function(
                    "ppe.nn.parallel.DistributedDataParallel.synchronize"):
                params = dict(self.named_parameters())
                if self._negotiate_grads:
                    # find parameters that have gradients
                    has_grads = torch.tensor(
                        [params[name].grad is not None
                         for name in self._sorted_param_keys],
                        device=self._device
                    )

                    # cast to long because bool may not be used in all_reduce
                    has_grads = has_grads.long()
                    with record(
                        "pytorch_pfn_extras.nn.parallel."
                        "DistributedDataParallel:coordinate",
                        use_cuda=torch.cuda.is_available(),
                    ):
                        dist.all_reduce(  # type: ignore[no-untyped-call]
                            has_grads, op=dist.ReduceOp.MAX)

                    for name, has_grad in zip(self._sorted_param_keys,
                                              has_grads.bool().cpu()):
                        # create zero tensor as a gradient if a parameter
                        # does not have the gradient and other processes
                        # require to synchronize this parameter.
                        if has_grad and params[name].grad is None:
                            params[name].grad = \
                                torch.zeros_like(params[name].data)

                grads = [params[name].grad for name in self._sorted_param_keys
                         if params[name].grad is not None]
                groups = _group_by_type(grads)
                with record(
                    "pytorch_pfn_extras.nn.parallel."
                    "DistributedDataParallel:reduce_gradient",
                    use_cuda=torch.cuda.is_available(),
                ):
                    for group in groups:
                        self._reduce_function(group, self._process_group)

                if self._broadcast_buffers:
                    buffers = dict(self.named_buffers())
                    bufs = [buffers[name] for name in self._sorted_buffer_keys]
                    groups = _group_by_type(bufs)
                    with record(
                        "pytorch_pfn_extras.nn.parallel."
                        "DistributedDataParallel:broadcast_buffer",
                        use_cuda=torch.cuda.is_available(),
                    ):
                        for group in groups:
                            self._broadcast_function(
                                group, self._process_group)

        # PyTorch will invoke `_synchronize` after the backward computation.
        Variable._execution_engine.queue_callback(_synchronize)

    def _input_to_device(self, obj: Any) -> Any:
        """Send data to the target device

        Analogous to `torch.nn.parallel.scatter_gather.scatter`
        """
        if isinstance(obj, torch.Tensor):
            if not self._show_send_gpu_warning and obj.device != self._device:
                logger.warning(
                    "Data are moved from {}"
                    " to {}".format(obj.device, self._device)
                )
                self._show_send_gpu_warning = True
            return obj.to(self._device)
        if isinstance(obj, tuple) and len(obj) > 0:
            return tuple([self._input_to_device(x) for x in obj])
        if isinstance(obj, list) and len(obj) > 0:
            return [self._input_to_device(x) for x in obj]
        if isinstance(obj, dict) and len(obj) > 0:
            return {key: self._input_to_device(value)
                    for key, value in obj.items()}
        return obj
