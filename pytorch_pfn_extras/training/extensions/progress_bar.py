import datetime
import sys

from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training.extensions import util


class ProgressBar(extension.Extension):

    """An extension to print a progress bar and recent training status.

    This extension prints a progress bar at every call. It watches the current
    iteration and epoch to print the bar.

    Args:
        training_length (tuple or None): Length of whole training. It consists
            of an integer and either ``'epoch'`` or ``'iteration'``. If this
            value is omitted and the stop trigger of the manager is
            :class:`IntervalTrigger`, this extension uses its attributes to
            determine the length of the training.
        update_interval (int): Number of iterations to skip printing the
            progress bar.
        bar_length (int): Length of the progress bar in characters.
        out: Stream to print the bar. Standard output is used by default.

    """

    def __init__(self, training_length=None, update_interval=100,
                 bar_length=50, out=sys.stdout):
        self._training_length = training_length
        self._update_interval = update_interval
        self._bar_length = bar_length
        self._out = out
        self._pbar = _ManagerProgressBar(
            self._training_length, self._bar_length, self._out)

    def __call__(self, manager):
        if self._pbar.manager is None:
            self._pbar.manager = manager

        iteration = manager.iteration
        # print the progress bar
        if iteration % self._update_interval == 0:
            self._pbar.update()

    def finalize(self):
        self._pbar.close()


class _ManagerProgressBar(util.ProgressBar):

    def __init__(self, training_length, bar_length, out):
        super().__init__(out)
        self.training_length = training_length
        self.bar_length = bar_length
        self.progress_template = None
        self.manager = None

    def get_lines(self):
        assert self.manager is not None
        lines = []

        iteration = self.manager.iteration
        epoch = self.manager.epoch_detail

        if self.training_length is None:
            t = self.manager._stop_trigger
            self.training_length = t.get_training_length()
        length, unit = self.training_length

        if unit == 'iteration':
            rate = iteration / length
        else:
            rate = epoch / length
        rate = min(rate, 1.0)

        bar_length = self.bar_length
        marks = '#' * int(rate * bar_length)
        lines.append('     total [{}{}] {:6.2%}\n'.format(
            marks, '.' * (bar_length - len(marks)), rate))

        epoch_rate = epoch - int(epoch)
        marks = '#' * int(epoch_rate * bar_length)
        lines.append('this epoch [{}{}] {:6.2%}\n'.format(
            marks, '.' * (bar_length - len(marks)), epoch_rate))

        if self.progress_template is None:
            self.progress_template = (
                '{0.iteration:10} iter, {0.epoch} epoch / %s %ss\n' %
                self.training_length)
        progress = self.progress_template.format(self.manager)
        lines.append(progress)

        speed_t, speed_e = self.update_speed(iteration, epoch)
        if unit == 'iteration':
            estimated_time = (length - iteration) / speed_t
        else:
            estimated_time = (length - epoch) / speed_e
        estimated_time = max(estimated_time, 0.0)
        lines.append('{:10.5g} iters/sec. Estimated time to finish: {}.\n'
                     .format(speed_t,
                             datetime.timedelta(seconds=estimated_time)))

        return lines
