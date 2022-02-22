from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training import triggers
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol


if TYPE_CHECKING:
    from pytorch_pfn_extras.training._trigger_util import TriggerLike


class BestValue(extension.Extension):

    """Extension traces the best value of a specific key in the observation.

    Args:
        key (str): Key of value.
        compare (callable): Compare function which takes current best value and
            new value and returns whether new value is better than current
            best.
        trigger: Trigger that decides the comparison interval between current
            best value and new value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~pytorch_pfn_extras.triggers.BestValueTrigger`.
    """

    default_name = 'best_value'

    def __init__(
            self,
            key: str,
            compare: Callable[[float, float], bool],
            trigger: 'TriggerLike' = (1, 'epoch'),
    ) -> None:
        self._best_epoch: Optional[int] = None
        self._best_it: Optional[int] = None
        self._best_trigger = triggers.BestValueTrigger(key, compare, trigger)

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        if self._best_trigger(manager):
            self._best_it, self._best_epoch = manager.iteration, manager.epoch

    def _check_best_value_exists(self) -> None:
        if self._best_trigger._best_value is None:
            raise RuntimeError("Best observation hasn't been obtained. "
                               "Run the BestValue extension at least once")

    @property
    def best_value(self) -> float:
        """Returns the current best value.

        If no value has been observed yet, it raises a RuntimError.
        """
        self._check_best_value_exists()
        return self._best_trigger._best_value  # type: ignore[return-value]

    @property
    def best_iteration(self) -> int:
        """Returns the iteration count that the current best value is observed.

        If no value has been observed yet, it raises a RuntimError.
        """
        self._check_best_value_exists()
        return self._best_it  # type: ignore[return-value]

    @property
    def best_epoch(self) -> int:
        """Returns the epoch count that the current best value is observed.

        If no value has been observed yet, it raises a RuntimError.
        """
        self._check_best_value_exists()
        return self._best_epoch  # type: ignore[return-value]

    def state_dict(self) -> Dict[str, Any]:
        return {
            '_best_trigger': self._best_trigger.state_dict(),
            '_best_it': self._best_it,
            '_best_epoch': self._best_epoch
        }

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        self._best_trigger.load_state_dict(to_load['_best_trigger'])
        self._best_it = to_load['_best_it']
        self._best_epoch = to_load['_best_epoch']


class MaxValue(BestValue):

    """Extension traces the maximum value of a specific key in the observation.

    Args:
        key (str): Key of value.
        trigger: Trigger that decides the comparison interval between current
            maximum value and new value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~pytorch_pfn_extras.triggers.BestValueTrigger`.
    """

    default_name = 'max_value'

    def __init__(self, key: str, trigger: 'TriggerLike' = (1, 'epoch')):
        super().__init__(
            key, lambda max_value, new_value: new_value > max_value, trigger)


class MinValue(BestValue):

    """Extension traces the maximum value of a specific key in the observation.

    Args:
        key (str): Key of value.
        trigger: Trigger that decides the comparison interval between current
            maximum value and new value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~pytorch_pfn_extras.triggers.BestValueTrigger`.
    """

    default_name = 'min_value'

    def __init__(self, key: str, trigger: 'TriggerLike' = (1, 'epoch')):
        super().__init__(
            key, lambda min_value, new_value: new_value < min_value, trigger)
