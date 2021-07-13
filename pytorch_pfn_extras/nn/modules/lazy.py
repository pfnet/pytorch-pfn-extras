import inspect
from typing import Tuple
import warnings

import torch


class LazyInitializationMixin:

    """A mixin for modules that lazily initialize buffers and parameters.

    Unlike regular modules, subclasses of this module can initialize
    buffers and parameters outside of the constructor (``__init__``).
    This allows you to, for example, initialize parameters in ``forward``
    method to determine the shape of the weight based on the initial input.

    Be sure to run "dummy" forward once to initialize all parameters that
    should be trained, before passing ``module.parameters()`` to an optimizer;
    otherwise weights initialized after ``module.parameters()`` (e.g., in
    ``forward`` function) will never be trained.

    Note that lazy modules cannot validate if the shape is correct during
    deserialization.  Also note that the initial weights may become different
    from the original (non-lazy) module even if the random seed is manually
    configured, as the order of initialization is different from the original
    one; especially, ``module.cuda()`` may cause the initialization to run on
    a GPU.

    The default value of lazy buffers and parameters are ``torch.Tensor([])``
    and ``UninitializedParameter()``, respectively.
    """

    # Subclasses must override these fields and list names of all buffers /
    # parameters that will be initialized lazily.
    lazy_buffer_names: Tuple[str, ...] = ()
    lazy_parameter_names: Tuple[str, ...] = ()

    def __init__(self, *args, **kwargs):
        self._lazy_ready = False

        super().__init__(*args, **kwargs)

        for name in self.lazy_buffer_names:
            self.register_buffer(name, torch.Tensor([]))
        for name in self.lazy_parameter_names:
            self.register_parameter(name, UninitializedParameter())
        self._register_load_state_dict_pre_hook(self._lazy_load_hook)
        self._lazy_ready = True

    @property
    def lazy_parmeters_determined(self):
        """Returns if all lazy parameters are determined.

        Subclasses can perform parameters initialization after all lazy
        parameters are determined.  Note that this may be called during
        ``__init__``.
        """
        return self._lazy_ready and all([
            not isinstance(getattr(self, x), UninitializedParameter)
            for x in self.lazy_parameter_names])

    def state_dict(self, *args, **kwargs):
        """Returns a dictionary containing a whole state of the module.

        This function overrides the default behavior to exclude uninitialized
        parameter from serialization.  This is needed because we need to
        discriminate lazy parameters (``UninitializedParameter()`) and
        initialized empty parameters (``torch.nn.Parameter(torch.Tensor())``)
        during deserialization.

        See comments of ``_lazy_load_hook`` for details.
        """
        destination = super().state_dict(*args, **kwargs)
        for name in self.lazy_parameter_names:
            if isinstance(getattr(self, name), UninitializedParameter):
                del destination[name]
        return destination

    def _lazy_load_hook(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        """load_state_dict pre-hook function for lazy buffers and parameters.

        The purpose of this hook is to check the current state and/or
        ``state_dict`` being loaded and ensure that both are states
        are properly initialized.

        See comment in ``torch.nn.Module._register_load_state_dict_pre_hook``
        for the details of the hook specification.
        """
        for name in self.lazy_buffer_names:
            key = prefix + name
            module_initialized = getattr(self, name).shape != (0,)
            state_initialized = state_dict[key].shape != (0,)
            if module_initialized and not state_initialized:
                raise RuntimeError(
                    'Can\'t load non-initialized buffers in already '
                    'initialized modules')
            elif not module_initialized and state_initialized:
                # Here we need to avoid a tensor size mismatch
                # this is a regular tensor without a materialize
                # method, so we can just resize for the load logic to copy
                # the contents later to the correct device the module
                # was moved to
                getattr(self, name).resize_(state_dict[key].size())

        for name in self.lazy_parameter_names:
            # The parameter does not exist in the loaded ``state_dict`` if the
            # original module was serialized before initializing lazy
            # parameters (see comments of ``state_dict``).
            key = prefix + name
            module_initialized = not isinstance(
                getattr(self, name), UninitializedParameter)
            state_initialized = key in state_dict
            if module_initialized and not state_initialized:
                raise RuntimeError(
                    'Can\'t load uninitialized parameters in already '
                    'initialized modules')
            elif not module_initialized and state_initialized:
                getattr(self, name).materialize(state_dict[key].shape)
            elif key not in state_dict and not module_initialized:
                param = UninitializedParameter()
                state_dict[key] = param


class UninitializedParameter(torch.nn.Parameter):

    def __repr__(self):
        return 'Uninitialized lazy parameter'

    def share_memory_(self):
        raise RuntimeError(
            'Can\'t share memory on an unitialized parameter. '
            'Run forward to initialize the network before calling '
            '`module.share_memory()`.')

    @property
    def is_leaf(self):
        # Hacky workaround to detect use of uninitialized lazy parameters.
        # This overrides ``is_leaf`` attribute which should always be ``True``
        # for parameters; optimizers check for this attribute and raise an
        # error if non-leaf tensors are detected.
        frame = inspect.currentframe()
        if frame.f_back.f_globals['__package__'].startswith('torch.optim'):
            warnings.warn('''
    Use of uninitialized lazy parameter in Optimizer has been detected.
    Maybe you forgot to run forward before passing `module.parameters()` to the optimizer?''')  # NOQA
        return True

    def materialize(self, shape, device=None, dtype=None):
        r"""Create a Parameter with the same properties of the uninitialized
        one. Given a shape, it materializes a parameter in the same device
        and with the same `dtype` as the current one or the specified ones in
        the arguments.

        Args:
            shape : (tuple): the shape for the materialized tensor.
            device (:class:`torch.device`): the desired device of the
                parameters
                and buffers in this module. Optional.
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module.
                Optional.
        """
        if device is None:
            device = self.data.device
        if dtype is None:
            dtype = self.data.dtype
        self.data = torch.empty(shape, device=device, dtype=dtype)
        self.__class__ = torch.nn.Parameter
