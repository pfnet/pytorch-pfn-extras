from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from pytorch_pfn_extras import reporting
from pytorch_pfn_extras.training import trigger as trigger_module
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol


if TYPE_CHECKING:
    from pytorch_pfn_extras.training._trigger_util import TriggerLike


class BestValueTrigger(trigger_module.Trigger):

    """Trigger invoked when specific value becomes best.

    Args:
        key (str): Key of value.
        compare (callable): Compare function which takes current best value and
            new value and returns whether new value is better than current
            best.
        trigger: Trigger that decides the comparison interval between current
            best value and new value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~pytorch_pfn_extras.triggers.IntervalTrigger`.

    """

    def __init__(
            self,
            key: str,
            compare: Callable[[float, float], bool],
            trigger: 'TriggerLike' = (1, 'epoch'),
    ) -> None:
        self._key = key
        self._best_value: Optional[float] = None
        self._interval_trigger = trigger_module.get_trigger(trigger)
        self._init_summary()
        self._compare = compare

    def __call__(self, manager: ExtensionsManagerProtocol) -> bool:
        """Decides whether the extension should be called on this iteration.

        Args:
            manager (~pytorch_pfn_extras.training.ExtensionsManager):
                Manager object that this
                trigger is associated with. The ``observation`` of this manager
                is used to determine if the trigger should fire.

        Returns:
            bool: ``True`` if the corresponding extension should be invoked in
            this iteration.

        """

        observation = manager.observation
        summary = self._summary
        key = self._key
        if key in observation:
            summary.add({key: observation[key]})

        if not self._interval_trigger(manager):
            return False

        stats = summary.compute_mean()
        if key not in stats:
            raise KeyError('Key "{}" not found in the observation '
                           '(current available keys are {})'
                           .format(key, list(stats.keys())))

        value = float(stats[key])  # copy to CPU
        self._init_summary()

        if self._best_value is None or self._compare(self._best_value, value):
            self._best_value = value
            return True
        return False

    def _init_summary(self) -> None:
        self._summary = reporting.DictSummary()

    def state_dict(self) -> Dict[str, Any]:
        state = {'interval_trigger': self._interval_trigger.state_dict(),
                 '_summary': self._summary.state_dict(),
                 '_best_value': self._best_value}
        return state

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        self._interval_trigger.load_state_dict(to_load['interval_trigger'])
        self._summary.load_state_dict(to_load['_summary'])
        self._best_value = to_load['_best_value']

    def may_fire(self, iteration: int, epoch_length: int) -> bool:
        return self._interval_trigger.may_fire(iteration, epoch_length)


class MaxValueTrigger(BestValueTrigger):

    """Trigger invoked when specific value becomes maximum.

    For example you can use this trigger to take snapshot on the epoch the
    validation accuracy is maximum.

    Args:
        key (str): Key of value. The trigger fires when the value associated
            with this key becomes maximum.
        trigger: Trigger that decides the comparison interval between current
            best value and new value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~pytorch_pfn_extras.triggers.IntervalTrigger`.

    """

    def __init__(self, key: str, trigger: 'TriggerLike' = (1, 'epoch')):
        super().__init__(
            key, lambda max_value, new_value: new_value > max_value, trigger)


class MinValueTrigger(BestValueTrigger):

    """Trigger invoked when specific value becomes minimum.

    For example you can use this trigger to take snapshot on the epoch the
    validation loss is minimum.

    Args:
        key (str): Key of value. The trigger fires when the value associated
            with this key becomes minimum.
        trigger: Trigger that decides the comparison interval between current
            best value and new value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~pytorch_pfn_extras.triggers.IntervalTrigger`.

    """

    def __init__(self, key: str, trigger: 'TriggerLike' = (1, 'epoch')):
        super().__init__(
            key, lambda min_value, new_value: new_value < min_value, trigger)
