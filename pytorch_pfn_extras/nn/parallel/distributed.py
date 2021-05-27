import logging

from contextlib import contextmanager
from collections import OrderedDict

import torch
from torch import nn
from torch import distributed as dist
from torch.utils import hooks
from torch.autograd import Variable
from torch.autograd.profiler import record_function
import threading

logger = logging.getLogger(__name__)


class _ApexWrapper:
    """A wrapper class to use nvidia/apex.

    Apex contains highly optimized functions. However, some environments may
    not install apex.
    This class provides fallback functions to handle such cases.
    """
    def __init__(self):
        try:
            import apex_C
            self.flatten = apex_C.flatten
            self.unflatten = apex_C.unflatten
        except ImportError:
            logger.warning(
                "fail to import apex_C: "
                "apex was not installed or installed without --cpp_ext.")
            self.flatten = torch._utils._flatten_dense_tensors
            self.unflatten = torch._utils._unflatten_dense_tensors

        try:
            import amp_C
            self._amp_C_multi_tensor_scale = amp_C.multi_tensor_scale
        except ImportError:
            logger.warning(
                "fail to import amp_C: "
                "apex was not installed or installed without --cpp_ext.")
            self._amp_C_multi_tensor_scale = None

    def multi_tensor_scale(self, src, dst, scale):
        if self._amp_C_multi_tensor_scale is None:
            self._multi_tensor_scale(src, dst, scale)
            return

        dtype = src[0].dtype
        device = src[0].device
        if (dtype == torch.float or dtype == torch.float16
                or dtype == torch.float64) and device.type == "cuda":
            self._multi_tensor_scale_apex(src, dst, scale)
        else:
            self._multi_tensor_scale(src, dst, scale)

    def _multi_tensor_scale_apex(self, src, dst, scale):
        overflow_buf = torch.tensor([0], device=src[0].device).int()
        self._amp_C_multi_tensor_scale(65536, overflow_buf, [src, dst],
                                       scale)

    def _multi_tensor_scale(self, src, dst, scale):
        with torch.no_grad():
            for s, d in zip(src, dst):
                d.copy_(s * scale)


apex_wrapper = None
_apex_wrapper_mutex = threading.Lock()


def get_apex_wrapper():
    global apex_wrapper
    if apex_wrapper is None:
        with _apex_wrapper_mutex:
            if apex_wrapper is None:
                apex_wrapper = _ApexWrapper()
    return apex_wrapper


def _reduce(values, group):
    size = sum([v.numel() for v in values])

    # flatten values to improve the runtime perfomance of all-reduce
    coalesced = torch.empty(size, device=values[0].device,
                            dtype=values[0].dtype)
    coalesced_views = get_apex_wrapper().unflatten(coalesced, values)
    get_apex_wrapper().multi_tensor_scale(values, coalesced_views, 1.0)

    with record_function("torch.distributed.all_reduce"):
        dist.all_reduce(coalesced, group=group)

    # unflatten values
    get_apex_wrapper().multi_tensor_scale(
        coalesced_views, values,
        1.0 / dist.get_world_size(group)
    )


def _broadcast(values, group):
    with torch.no_grad():
        coalesced = get_apex_wrapper().flatten(values)
        with record_function("torch.distributed.broadcast"):
            dist.broadcast(coalesced, 0, group=group)
        get_apex_wrapper().multi_tensor_scale(
            get_apex_wrapper().unflatten(coalesced, values),
            values, 1.0
        )


def _group_by_type(values):
    groups = {}
    for value in values:
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

    def __init__(self, module,
                 broadcast_buffers=True,
                 negotiate_grads=True,
                 process_group=None,
                 reduce_function=None,
                 broadcast_function=None,
                 **kwargs):
        super().__init__()

        """
        This module receives keyword arguments for the compatibility with
        `torch.nn.parallel.DistributedDataParallel`.
        It shows a warning when setting the ignored arguments.
        """
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

        self._comm_hooks = OrderedDict()

        self._require_sync = True
        self._show_send_gpu_warning = False

        # synchronize initial parameters and buffers
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        values = \
            [params[name] for name in self._sorted_param_keys] + \
            [buffers[name] for name in self._sorted_buffer_keys]
        if dist.is_initialized():
            groups = _group_by_type(values)
            for group in groups:
                _broadcast(group, self._process_group)
        else:
            logger.warning("torch.distributed is not initialized")

        # add hook to launch synchronization
        self.register_backward_hook(self._backward_hook)

    @contextmanager
    def no_sync(self):
        """A context manager to disable synchronization after backward
        """
        prev = self._require_sync
        self._require_sync = False
        try:
            yield
        finally:
            self._require_sync = prev

    def forward(self, *args, **kwargs):
        args = self._input_to_device(args)
        kwargs = self._input_to_device(kwargs)
        return self.module(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.module.load_state_dict(*args, **kwargs)

    def state_dict(self):
        return self.module.state_dict()

    def register_comm_hook(self, hook):
        """Registers a hook function. This module will invoke the hook before
        starting the synchronization.

        Args:
        hook: Callable object that will be invoked before synchronization
        """
        handle = hooks.RemovableHandle(self._comm_hooks)
        self._comm_hooks[handle.id] = hook
        return handle

    def _backward_hook(self, module, gin, gout):
        def _synchronize():
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
                    dist.all_reduce(has_grads, op=dist.ReduceOp.MAX)

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
                for group in groups:
                    self._reduce_function(group, self._process_group)

                if self._broadcast_buffers:
                    buffers = dict(self.named_buffers())
                    bufs = [buffers[name] for name in self._sorted_buffer_keys]
                    groups = _group_by_type(bufs)
                    for group in groups:
                        self._broadcast_function(group, self._process_group)

        # PyTorch will invoke `_synchronize` after the backward computation.
        Variable._execution_engine.queue_callback(_synchronize)

    def _input_to_device(self, obj):
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
