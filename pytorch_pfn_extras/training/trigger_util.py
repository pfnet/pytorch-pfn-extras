class Trigger:
    """Base class for triggers."""
    def load_state_dict(self, state):
        pass

    def state_dict(self):
        return {}

    def __call__(self, manager):
        raise NotImplementedError


def get_trigger(trigger):
    """Gets a trigger object.

    Trigger object is a callable that accepts a
    :class:`~pytorch_pfn_extras.training.ExtensionsManager` object
    as an argument and returns a boolean value.
    When it returns True, various kinds of events can occur
    depending on the context in which the trigger is used. For example, if the
    trigger is passed to the
    :meth:`~pytorch_pfn_extras.training.ExtensionsManager.extend` method of
    a manager, then the registered extension is invoked only when the trigger
    returns True.

    This function returns a trigger object based on the argument.
    If ``trigger`` is already a callable, it just returns the trigger. If
    ``trigger`` is ``None``, it returns a trigger that never fires. Otherwise,
    it creates a :class:`~pytorch_pfn_extras.triggers.IntervalTrigger`.

    Args:
        trigger: Trigger object. It can be either an already built trigger
            object (i.e., a callable object that accepts a manager object and
            returns a bool value), or a tuple. In latter case, the tuple is
            passed to :class:`~pytorch_pfn_extras.triggers.IntervalTrigger`.

    Returns:
        ``trigger`` if it is a callable, otherwise a
        :class:`~pytorch_pfn_extras.triggers.IntervalTrigger`
        object made from ``trigger``.

    """
    from pytorch_pfn_extras.training.triggers import interval_trigger

    if callable(trigger):
        return trigger
    elif trigger is None:
        return _never_fire_trigger
    else:
        return interval_trigger.IntervalTrigger(*trigger)


def _never_fire_trigger(manager):
    return False
