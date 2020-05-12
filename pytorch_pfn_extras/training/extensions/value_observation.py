from pytorch_pfn_extras.training import extension


def observe_value(observation_key, target_func):
    """Returns an extension to continuously record a value.

    Args:
        observation_key (str): Key of observation to record.
        target_func (function): Function that returns the value to record.
            It must take one argument:
            :class:~pytorch_pfn_extras.training.ExtensionsManager object.
    Returns:
        The extension function.

    This extension is triggered each epoch by default.
    To change this, use the ``trigger`` argument with the
    :meth:`ExtensionsManager.extend() <pytorch_pfn_extras.training\
           .ExtensionsManager>` method.

    """
    @extension.make_extension(
        trigger=(1, 'epoch'), priority=extension.PRIORITY_WRITER)
    def _observe_value(manager):
        manager.observation[observation_key] = target_func(manager)
    return _observe_value


def observe_lr(optimizer, param_group=0, observation_key='lr'):
    """Returns an extension to record the learning rate.

    Args:
        optimizer (Optimizer): Optimizer whose learning rate is
            recorded.
        param_group (int): Param group of the optimizer to observe
        observation_key (str): Key of observation to record.

    Returns:
        The extension function.

    This extension is triggered each epoch by default.
    To change this, use the ``trigger`` argument with the
    :meth:`ExtensionsManager.extend() <pytorch_pfn_extras.training\
           .ExtensionsManager>` method.

    """
    return observe_value(
        observation_key,
        lambda manager: optimizer.param_groups[param_group]['lr'])
