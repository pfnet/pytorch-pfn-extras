import operator
from typing import Tuple, TYPE_CHECKING
import warnings

from pytorch_pfn_extras import reporting
from pytorch_pfn_extras.training import trigger
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol


if TYPE_CHECKING:
    from pytorch_pfn_extras.training._trigger_util import TriggerLike
    from pytorch_pfn_extras.training._trigger_util import UnitLiteral


class EarlyStoppingTrigger(trigger.Trigger):
    """__init__(\
        self, check_trigger=(1, 'epoch'), monitor='main/loss', \
        patience=3, mode='auto', verbose=False, \
        max_trigger=(100, 'epoch'))

    Trigger for Early Stopping

    This trigger works as follows.
    Within each *check interval* defined by the ``check_trigger`` argument,
    it monitors and accumulates the reported value at each iteration.
    At the end of each interval, it computes the mean of the accumulated
    values and compares it to the previous ones to maintain the *best* value.
    When it finds that the best value is not updated
    for some periods (defined by ``patience``), this trigger fires.

    Args:
        monitor (str) : The metric you want to monitor
        check_trigger: Trigger that decides the comparison
            interval between current best value and new value.
            This must be a tuple in the form of ``<int>,
            'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~pytorch_pfn_extras.triggers.IntervalTrigger`.
        patience (int) : Counts to let the trigger be patient.
            The trigger will not fire until the condition is met
            for successive ``patience`` checks.
        mode (str) : ``'max'``, ``'min'``, or ``'auto'``.
            It is used to determine how to compare the monitored values.
        verbose (bool) : Enable verbose output.
            If verbose is true, you can get more information
        max_trigger: Upper bound of the number of training loops
    """

    def __init__(
            self,
            check_trigger: 'TriggerLike' = (1, 'epoch'),
            monitor: str = 'main/loss',
            patience: int = 3,
            mode: str = 'auto',
            verbose: bool = False,
            max_trigger: Tuple[int, 'UnitLiteral'] = (100, 'epoch'),
    ) -> None:
        self.count = 0
        self.patience = patience
        self.monitor = monitor
        self.verbose = verbose
        self.already_warning = False
        self._max_trigger = trigger.IntervalTrigger(*max_trigger)
        self._interval_trigger = trigger.get_trigger(check_trigger)

        self._init_summary()

        if mode == 'max':
            self._compare = operator.gt

        elif mode == 'min':
            self._compare = operator.lt

        else:
            if 'accuracy' in monitor:
                self._compare = operator.gt

            else:
                self._compare = operator.lt

        if self._compare == operator.gt:
            if verbose:
                print('early stopping: operator is greater')
            self.best = float('-inf')

        else:
            if verbose:
                print('early stopping: operator is less')
            self.best = float('inf')

    def __call__(self, manager: ExtensionsManagerProtocol) -> bool:
        """Decides whether the training loop should be stopped.

        Args:
            manager (~pytorch_pfn_extras.training.ExtensionsManager):
                Manager object that this
                trigger is associated with. The ``observation`` of this manager
                is used to determine if the trigger should fire.

        Returns:
            bool: ``True`` if the training loop should be stopped.
        """

        observation = manager.observation

        summary = self._summary

        if self.monitor in observation:
            summary.add({self.monitor: observation[self.monitor]})

        if self._max_trigger(manager):
            return True

        if not self._interval_trigger(manager):
            return False

        if self.monitor not in observation.keys():
            warnings.warn('{} is not in observation'.format(self.monitor))
            return False

        stat = self._summary.compute_mean()
        current_val = float(stat[self.monitor])
        self._init_summary()

        if self._compare(current_val, self.best):
            self.best = current_val
            self.count = 0

        else:
            self.count += 1

        if self._stop_condition():
            if self.verbose:
                print('Epoch {}: early stopping'.format(manager.epoch))
            return True

        return False

    def _stop_condition(self) -> bool:
        return self.count >= self.patience

    def _init_summary(self) -> None:
        self._summary = reporting.DictSummary()

    def get_training_length(self) -> Tuple[float, str]:
        return self._max_trigger.get_training_length()

    def may_fire(self, iteration: int, epoch_length: int) -> bool:
        return self._interval_trigger.may_fire(iteration, epoch_length)
