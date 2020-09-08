import inspect
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
    lazy_buffer_names = ()
    lazy_parameter_names = ()

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

        The purpose of this hook is to adjust the current state and/or
        ``state_dict`` being loaded so that a module instance serialized in
        both un/initialized state can be deserialized onto both un/initialized
        module instance.

        See comment in ``torch.nn.Module._register_load_state_dict_pre_hook``
        for the details of the hook specification.
        """
        for name in self.lazy_buffer_names:
            # Avoid shape mismatch error when loading an initialized buffer
            # onto an uninitialized module instance.
            self.register_buffer(name, state_dict[prefix + name])

        for name in self.lazy_parameter_names:
            # The parameter may not exist in the loaded ``state_dict`` if the
            # original module was serialized before initializing lazy
            # parameters (see comments of ``state_dict``).
            key = prefix + name
            if key in state_dict:
                # The model was serialized after initialization.
                self.register_parameter(
                     name, torch.nn.Parameter(state_dict[key]))
            else:
                # The model was serialized before initialization.
                param = UninitializedParameter()
                self.register_parameter(name, param)
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
