from typing import Any, Callable, Dict, Union, Optional, Tuple


class TriggerArgument:
    @property
    def iteration(self) -> int:
        raise NotImplementedError

    @property
    def epoch(self) -> int:
        raise NotImplementedError

    @property
    def epoch_detail(self) -> float:
        raise NotImplementedError

    @property
    def elapsed_time(self) -> float:
        raise NotImplementedError


# TODO: Use `Literal['epoch', 'iteration']` after Py3.7 is dropped
UnitLiteral = str
TriggerFunc = Callable[[TriggerArgument], bool]
TriggerLike = Optional[Union[TriggerFunc, Tuple[int, UnitLiteral]]]


class Trigger:
    """Base class for triggers."""
    def may_fire(self, iteration: int, epoch_len: int) -> bool:
        """Flags if the trigger may fire at the current iteration

        This must not alter the trigger state

        """
        return True

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def __call__(self, manager: TriggerArgument) -> bool:
        raise NotImplementedError


class _CallableTrigger(Trigger):
    def __init__(self, func: TriggerFunc) -> None:
        self.func = func

    def __call__(self, manager: TriggerArgument) -> bool:
        return self.func(manager)


def get_trigger(trigger: TriggerLike) -> Trigger:
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

    if isinstance(trigger, Trigger):
        return trigger
    elif callable(trigger):
        return _CallableTrigger(trigger)
    elif trigger is None:
        return _CallableTrigger(_never_fire_trigger)
    else:
        return interval_trigger.IntervalTrigger(*trigger)


def _never_fire_trigger(manager: TriggerArgument) -> bool:
    return False
