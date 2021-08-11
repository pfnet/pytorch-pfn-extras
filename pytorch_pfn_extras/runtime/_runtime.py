import torch

_RUNTIME_TAG_NAME = '_ppe_runtime'


class BaseRuntime:
    """A base class for collections of device-specific callback functions.

    The function attributes of this class will be called from
    ``ppe.to`` or ``ppe.handler.Handler``.

    ``ppe.runtime.runtime_registry`` stores the runtime classes and
    dispatches them by feeding the corresponding name string as an input.

    Args:
        device_spec (torch.device or str):
            The device that modules and tensors are transferred to.
        config (dict):
            A configuration dictionary that can be used from runtime method.
    """

    def __init__(self, device_spec, config=None):
        self.device_spec = device_spec
        self.config = config

    def convert_batch(self, args):
        """Transfers the given batch to the specific device.

        Args:
            args (object): A batch data of any type.

        Returns:
            A batch data transferred to the specific device
            of the same type as input.
        """

        # this should be called with the runtime associated to a model
        # or a model part
        if isinstance(args, dict):
            return {
                k: self.move_tensor(v) if isinstance(v, torch.Tensor) else v
                for k, v in args.items()
            }
        elif isinstance(args, (list, tuple)):
            return [
                self.move_tensor(v) if isinstance(v, torch.Tensor) else v
                for v in args
            ]
        elif isinstance(args, torch.Tensor):
            return self.move_tensor(args)
        return args

    def move_module(self, module):
        """Transfers the module to the specific device.

        Before this method is called, ``ppe.to`` will add this class as
        an new attribute ("_ppe_runtime") to the input module.

        Args:
            module (torch.nn.Module): A module.

        Returns:
            A module transferred to the specific device.
        """
        raise NotImplementedError()

    def move_tensor(self, tensor):
        """Transfers the tensor to the specific device.

        Args:
            tensor (torch.Tensor): A tensor.

        Returns:
            A tensor transferred to the specific device.
        """
        raise NotImplementedError()

    def initialize_module(self, module, loader_or_batch, optimizer=None):
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

    def train_epoch_begin(self, module):
        """Preprocess of each epoch.

        Args:
            module (torch.nn.Module): A module.

        Returns: None
        """
        raise NotImplementedError()

    def train_pre_step(self, trainer, module, batch_idx, batch):
        """Preprocess of each step.

        This method is called at the beginning of every steps: the set of
        (typically one) iterations and an update.

        Args:
            trainer (_Trainer): A trainer.
            module (torch.nn.Module): A module.
            batch_idx (int): The batch index.
            batch (list of torch.Tensor):
                The list of input tensors of this batch.

        Returns: None
        """
        raise NotImplementedError()

    def train_post_step(self, trainer, module, batch_idx, batch, outs):
        """Postprocess of each step.

        This method is called at the end of every steps: the set of
        (typically one) iterations and an update.

        Args:
            trainer (_Trainer): A trainer.
            module (torch.nn.Module): A module.
            batch_idx (int): The batch index.
            batch (list of torch.Tensor):
                 The list of input tensors of this batch.
            outs: (list of torch.Tensor):
                 The list of output tensors of this batch.

        Returns: None
        """
        raise NotImplementedError()

    def train_validation_begin(self, module):
        """The method called before each evaluation.

        Args:
            module (torch.nn.Module): A module.

        Returns: None
        """
        raise NotImplementedError()

    def train_validation_end(self, module):
        """The method called after each evaluation.

        Args:
            module (torch.nn.Module): A module.

        Returns: None
        """
        raise NotImplementedError()

    def eval_pre_step(self, evaluator, module, batch_idx, batch):
        """The method called at the beginning of each evaluation.

        Args:
            evaluator (_Evaluator): An evaluator.
            module (torch.nn.Module): A module.
            batch_idx (int): The batch index.
            batch (list of torch.Tensor):
                 The list of input tensors of this batch.

        Returns: None
        """
        raise NotImplementedError()

    def eval_post_step(self, evaluator, module, batch_idx, batch, outs):
        """The method called at the end of each evaluation.

        Args:
            evaluator (_Evaluator): An evaluator.
            module (torch.nn.Module): A module.
            batch_idx (int): The batch index.
            batch (list of torch.Tensor):
                 The list of input tensors of this batch.
            outs: (list of torch.Tensor):
                 The list of output tensors of this batch.

        Returns: None
        """
        raise NotImplementedError()

    def get_pending_result(self, future, module, blocking):
        """The method called to retrieve the result of a asynchronous call.

        Args:
            module (torch.nn.Module): A module.
            future (object): An arbitrary object that the runtime can use
                for synchronization.
            blocking (bool): A flag to determine wether to wait for
                an asynchronous call completion or returns None.

        Returns: (list of torch.Tensor):
            The list of output tensors of the oldest async operation.
            Will return `None` if `blocking=False` and no asynchronous
            call is complete.
        """
        raise NotImplementedError()


class PyTorchRuntime(BaseRuntime):
    """A collections of callback functions for the devices that PyTorch
    supports by default.

    Args:
        device_spec (torch.device or str): The device.
    """

    def move_module(self, module):
        return module.to(self.device_spec)

    def move_tensor(self, tensor):
        return tensor.to(self.device_spec)

    def initialize_module(self, module, loader_or_batch, optimizer=None):
        pass

    def train_epoch_begin(self, module):
        pass

    def train_validation_begin(self, module):
        pass

    def train_validation_end(self, module):
        pass

    def train_pre_step(self, trainer, module, batch_idx, batch):
        pass

    def train_post_step(self, trainer, module, batch_idx, batch, outs):
        pass

    def eval_pre_step(self, evaluator, module, batch_idx, batch):
        pass

    def eval_post_step(self, evaluator, module, batch_idx, batch, outs):
        pass

    def get_pending_result(self, future, module, blocking):
        pass


def _module_runtime_tag(module):
    return getattr(module, _RUNTIME_TAG_NAME, None)


def _set_module_runtime_tag(module, runtime):
    return setattr(module, _RUNTIME_TAG_NAME, runtime)


def named_runtime_modules(module, module_name='',
                          first_level=True, recursive=True):
    # This can be invoked with no containarized modules
    # to look for submodules that hold containers
    if _module_runtime_tag(module) is None:
        if first_level or recursive:
            for name, sm in module.named_children():
                yield from named_runtime_modules(sm, name, False, recursive)
    else:
        yield module_name, module
