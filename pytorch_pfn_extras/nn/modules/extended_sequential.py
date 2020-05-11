import torch
import copy
from collections.abc import Iterable


def _reset_parameters(model):
    if isinstance(model, torch.nn.Sequential) or \
       isinstance(model, torch.nn.ModuleList):
        for submodel in model:
            _reset_parameters(submodel)
    elif isinstance(model, torch.nn.ModuleDict):
        for submodel in model.values():
            _reset_parameters(submodel)
    else:
        if isinstance(model, torch.nn.Module):
            model.reset_parameters()
    return model


class ExtendedSequential(torch.nn.Sequential):
    """Sequential module with extended features from chainer.

    """
    def _copy_model(self, mode):
        if mode == 'init':
            return _reset_parameters(copy.deepcopy(self))
        elif mode == 'copy':
            return copy.deepcopy(self)
        else:
            # mode == share
            return copy.copy(self)

    def repeat(self, n_repeat: int, mode: 'str' = 'init'):
        """Repeats this Sequential multiple times.

        This method returns a :class:`~torch.nn.Sequential` object which has
        original `Sequential` multiple times repeatedly. The ``mode``
        argument means how to copy this sequential to repeat.

        The functions is supposed to behave the same way as `repeat`
        in `chainer`.

        Args:
            n_repeat (int): Number of times to repeat.
            mode (str): It should be either ``copy``, or ``share``.
                ``copy`` means that the parameters will not be re-initialized
                but object itself will be deep-copied, so that all elements
                have same initial parameters but can be changed independently.
                ``share`` means all the elements which consist the resulting
                :class:`~torch.nn.Sequential` object are same object because
                they are shallow-copied, so that all parameters of elements
                are shared with each other.
                ``init`` is not supported yet.
        """
        if n_repeat <= 0:
            return ExtendedSequential()

        if mode not in ['copy', 'share', 'init']:
            raise ValueError(
                'The \'mode\' argument should be either ,'
                '\'copy\', or \'share\'. But {} was given.'.format(mode))

        model_list = []
        for _ in range(n_repeat):
            model_list.append(self._copy_model(mode))
        return ExtendedSequential(*model_list)
