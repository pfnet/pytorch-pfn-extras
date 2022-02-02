from typing import Any, Dict, Optional

from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol


class BestObservation(extension.Extension):

    """Traces the best observation value and its epoch, iterations.

    When this extension is triggerred, it checks the value of the specified key in
    the observations (``observations`` attribute in :class:`~ExtensionsManager`).
    If the value is better compared to that of previous call, it records
    that value, current epoch and iteration.

    For making observation and reporting, see :doc:`../../user_guide/reporting`
    and :doc:`../../user_guide/extensions`.

    Args:
        observation_key (str): Key storing the observation.
        direction (str): When ``'MINIMIZE'`` (default) it keeps the smallest
            observation, and when ``'MAXIMIZE'``, the largest.

    Attributes:
        best_value: The best value.
        best_iteration: The iteration count when the best value is observed.
        best_epoch: The epoch count when the best value is observed.

    """
    trigger = 1, 'epoch'
    default_name = 'best_observation'
    priority = extension.PRIORITY_EDITOR

    def __init__(
            self,
            observation_key: str,
            direction: str = 'MINIMIZE',
    ) -> None:
        self._key = observation_key

        self._direction = direction.upper()
        if self._direction not in ('MINIMIZE', 'MAXIMIZE'):
            raise ValueError('Direction "{}" not supported. '
                             'Please specify either "MINIMIZE" or "MAXIMIZE"'
                             .format(direction))

        self._best_epoch: Optional[int] = None
        self._best_it: Optional[int] = None
        self._best_value = float('inf') if self._direction == 'MINIMIZE' else float('-inf')

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        observation: Any = manager.observation

        if self._key not in observation:
            raise RuntimeError('Key "{}" not found in the observation '
                               '(current available keys are {})'
                               .format(self._key, list(observation.keys())))

        value = observation[self._key]
        if (self._direction == 'MINIMIZE' and value <= self._best_value) or \
           (self._direction == 'MAXIMIZE' and self._best_value <= value):
            self._best_value = value
            self._best_it, self._best_epoch = manager.iteration, manager.epoch

    def _check_best_value_exists(self) -> None:
        if not self._best_epoch:
            raise RuntimeError("Best observation hasn't been obtained. "
                               "Run the BestObservation extension at least once")

    @property
    def best_value(self) -> float:
        self._check_best_value_exists()
        return self._best_value  # type: ignore[return-value]

    @property
    def best_iteration(self) -> int:
        self._check_best_value_exists()
        return self._best_it  # type: ignore[return-value]

    @property
    def best_epoch(self) -> int:
        self._check_best_value_exists()
        return self._best_epoch  # type: ignore[return-value]

    def state_dict(self) -> Dict[str, Any]:
        return {
            '_direction': self._direction,
            '_best_value': self._best_value,
            '_best_iteration': self._best_it,
            '_best_epoch': self._best_epoch,
        }

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        self._direction = to_load['_direction']
        self._best_value = to_load['_best_value']
        self._best_it = to_load['_best_iteration']
        self._best_epoch = to_load['_best_epoch']
