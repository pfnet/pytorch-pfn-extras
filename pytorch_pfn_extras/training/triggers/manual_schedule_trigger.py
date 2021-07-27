from pytorch_pfn_extras.training import trigger


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

    Attributes:
        finished (bool): Flag that indicates whether or not this trigger will
        fire in the future. This flag is used to determine if the extension
        should be initialized after resume.

    """

    def __init__(self, points, unit):
        if unit not in ('epoch', 'iteration'):
            raise ValueError(
                'Trigger unit must be either \'epoch\' or \'iteration\'.')

        self.points = (points if isinstance(points, list) else [points])
        self.unit = unit
        self.finished = False

        self._previous_iteration = 0
        self._previous_epoch_detail = 0.

    def __call__(self, manager):
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
        if self.unit == 'epoch':
            epoch_detail = manager.epoch_detail
            previous_epoch_detail = self._previous_epoch_detail

            # if previous_epoch_detail is invalid value,
            # use the value of manager.
            if previous_epoch_detail < 0:
                previous_epoch_detail = manager.previous_epoch_detail

            fire = any(
                previous_epoch_detail < p <= epoch_detail
                for p in self.points)

            if hasattr(self, '_finished_is_tmp'):
                del self._finished_is_tmp
                if epoch_detail >= max(self.points):
                    self.finished = True
            if fire and epoch_detail >= max(self.points):
                self.finished = True
        else:
            iteration = manager.iteration
            previous_iteration = self._previous_iteration

            # if previous_iteration is invalid value,
            # guess it from current iteration.
            if previous_iteration < 0:
                previous_iteration = iteration - 1

            fire = any(
                previous_iteration < p <= iteration
                for p in self.points)

            if hasattr(self, '_finished_is_tmp'):
                del self._finished_is_tmp
                if iteration >= max(self.points):
                    self.finished = True
            if fire and iteration >= max(self.points):
                self.finished = True

        # save current values
        self._previous_iteration = manager.iteration
        if hasattr(manager, 'epoch_detail'):
            self._previous_epoch_detail = manager.epoch_detail

        return fire

    def state_dict(self):
        state = {}
        state['_previous_iteration'] = self._previous_iteration
        state['_previous_epoch_detail'] = self._previous_epoch_detail
        state['finished'] = self.finished
        return state

    def load_state_dict(self, to_load):
        self._previous_iteration = to_load['_previous_iteration']
        self._previous_epoch_detail = to_load['_previous_epoch_detail']
        self.finished = to_load['finished']
