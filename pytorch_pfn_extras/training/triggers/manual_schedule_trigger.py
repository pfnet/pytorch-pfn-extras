from typing import Sequence, Union, TYPE_CHECKING

from pytorch_pfn_extras.training import trigger
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol


if TYPE_CHECKING:
    from pytorch_pfn_extras.training._trigger_util import UnitLiteral


class ManualScheduleTrigger(trigger.Trigger):

    """Trigger invoked at specified point(s) of iterations or epochs.

    This trigger accepts iterations or epochs indicated by given point(s).
    There are two ways to specify the point(s): iteration and epoch.
    ``iteration`` means the number of updates, while ``epoch`` means the number
    of sweeps over the training dataset. Fractional values are allowed
    if the point is a number of epochs; the trigger uses the ``iteration``
    and ``epoch_detail`` attributes defined by the manager.

    Args:
        points (int, float, or list of int or float): time of the trigger.
            Must be an integer or list of integer if unit is ``'iteration'``.
        unit (str): Unit of the time specified by ``points``. It must be
            either ``'iteration'`` or ``'epoch'``.

    """

    def __init__(self, points: Union[float, Sequence[float]], unit: 'UnitLiteral'):
        if unit not in ('epoch', 'iteration'):
            raise ValueError(
                'Trigger unit must be either \'epoch\' or \'iteration\'.')

        self.points = (points if isinstance(points, list) else [points])
        self.unit = unit

    def __call__(self, manager: ExtensionsManagerProtocol) -> bool:
        """Decides whether the extension should be called on this iteration.

        Args:
            manager (~pytorch_pfn_extras.training.ExtensionsManager):
                Manager object that this trigger is associated with.
                The iteration information in this manager is used to
                determine if the trigger should fire.

        Returns:
            bool: True if the corresponding extension should be invoked in this
            iteration.

        """
        fire = self.may_fire(manager.iteration, manager._iters_per_epoch)
        return fire

    def may_fire(self, iteration: int, epoch_length: int) -> bool:
        if self.unit == 'epoch':
            fire = any(
                int(p * epoch_length) == iteration for p in self.points)
        else:
            fire = any(p == iteration for p in self.points)
        return fire
