from typing import Tuple, TYPE_CHECKING

from pytorch_pfn_extras.training import trigger
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol


if TYPE_CHECKING:
    from pytorch_pfn_extras.training._trigger_util import UnitLiteral


class IntervalTrigger(trigger.Trigger):

    """Trigger based on a fixed interval.

    This trigger accepts iterations divided by a given interval. There are two
    ways to specify the interval: per iterations and epochs. `Iteration` means
    the number of updates, while `epoch` means the number of sweeps over the
    training dataset. Fractional values are allowed if the interval is a
    number of epochs; the trigger uses the `iteration` and `epoch_detail`
    attributes defined by the manager.

    For the description of triggers see
    :func:`~pytorch_pfn_extras.get_trigger`.

    Args:
        period (int or float): Length of the interval. Must be an integer if
            unit is ``'iteration'``.
        unit (str): Unit of the length specified by ``period``. It must be
            either ``'iteration'`` or ``'epoch'``.

    """

    def __init__(self, period: float, unit: 'UnitLiteral'):
        if unit not in ('epoch', 'iteration'):
            raise ValueError(
                'Trigger unit must be either \'epoch\' or \'iteration\'.')
        self.period = period
        self.unit = unit

    def __call__(self, manager: ExtensionsManagerProtocol) -> bool:
        """Decides whether the extension should be called on this iteration.

        Args:
            manager (~pytorch_pfn_extras.training.ExtensionsManager):
                Manager object that this trigger is associated with.
                The iteration related information in this manager is used to
                determine if the trigger should fire.

        Returns:
            bool: True if the corresponding extension should be invoked in this
            iteration.

        """
        current_step = manager.iteration
        fire = self.may_fire(current_step, manager._iters_per_epoch)
        return fire

    def get_training_length(self) -> Tuple[float, str]:
        return (self.period, self.unit)

    def __str__(self) -> str:
        """Returns a string describing the class and interval

        Returns:
            str: IntervalTrigger(<period>, '<unit>')
        """
        return '{}({}, \'{}\')'.format(
            self.__class__.__name__, self.period, self.unit
        )

    def may_fire(self, iteration: int, epoch_length: int) -> bool:
        if iteration == 0:
            if self.unit == 'epoch':
                return epoch_length == 0
            else:
                return self.period == 0
        if self.unit == 'epoch':
            fire = (iteration % (epoch_length * self.period)) == 0
        else:
            fire = (iteration % self.period) == 0
        return fire
