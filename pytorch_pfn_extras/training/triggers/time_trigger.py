from pytorch_pfn_extras.training import trigger


class TimeTrigger(trigger.Trigger):

    """Trigger based on a fixed time interval.

    This trigger accepts iterations with a given interval time.

    Args:
        period (float): Interval time. It is given in seconds.

    """

    def __init__(self, period):
        self._period = period
        self._next_time = self._period

    def __call__(self, manager):
        if self._next_time < manager.elapsed_time:
            self._next_time += self._period
            return True
        else:
            return False

    def state_dict(self):
        state = {'next_time': self._next_time}
        return state

    def load_state_dict(self, to_load):
        self._next_time = to_load['next_time']
