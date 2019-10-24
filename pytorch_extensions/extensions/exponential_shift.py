from __future__ import division

from pytorch_extensions import extension


class ExponentialShift(extension.Extension):

    """Trainer extension to exponentially shift an optimizer attribute.

    This extension exponentially increases or decreases the specified attribute
    of the optimizer. The typical use case is an exponential decay of the
    learning rate.

    This extension is also called before the training loop starts by default.

    Args:
        attr (str): Name of the attribute to shift.
        rate (float): Rate of the exponential shift. This value is multiplied
            to the attribute at each call.
        optimizer (~torch.Optimizer): Target optimizer to adjust the
            attribute.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        target (float): Target value of the attribute. If the attribute reaches
            this value, the shift stops.
        param_group (int) : param_group of the optimizer to update the value

    """

    def __init__(
           self, attr, rate, optimizer, init=None, target=None, param_group=0):
        self._attr = attr
        if rate < 0:
            raise ValueError('ExponentialShift does not support negative rate')
        self._rate = rate
        self._init = init
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None
        self._param_group = param_group

    def initialize(self, trainer):
        optimizer = self._optimizer
        # ensure that _init is set
        if self._init is None:
            self._init = getattr(optimizer, self._attr)

        if self._last_value is not None:  # resuming from a snapshot
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._init)

    def __call__(self, manager):
        self._t += 1

        optimizer = self._optimizer
        value = self._init * (self._rate ** self._t)
        if self._target is not None:
            if self._rate > 1:
                # almost same as value = min(value, self._target), but this
                # line supports negative values, too
                if value / self._target > 1:
                    value = self._target
            else:
                # ditto
                if value / self._target < 1:
                    value = self._target
        self._update_value(optimizer, value)

    def serialize(self, serializer):
        # self._t = serializer('_t', self._t)
        # self._last_value = serializer('_last_value', self._last_value)
        # if isinstance(self._last_value, numpy.ndarray):
        #     self._last_value = self._last_value.item()
        pass

    def _update_value(self, optimizer, value):
        self._optimizer.param_groups[self._param_group][self._attr] = value

    def _get_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        return self._optimizer.param_groups[self._param_group][self._attr]
