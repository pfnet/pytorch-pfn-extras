import datetime
import sys
import time

from IPython.core.display import display
from ipywidgets import HTML, FloatProgress, HBox, VBox  # NOQA

from pytorch_pfn_extras.training import extension, trigger  # NOQA


class ProgressBarNotebook(extension.Extension):

    """An extension to print a progress bar and recent training status.

    It is aimed to work on jupyter notebook as replacement of `ProgressBar`.
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
        bar_length (int): This is not used, argument is kept to be consistent
            with `ProgressBar`.
        out: This is not used, argument is kept to be consistent with
            `ProgressBar`.

    """

    def __init__(self, training_length=None, update_interval=100,
                 bar_length=50, out=sys.stdout):
        self._training_length = training_length
        if training_length is not None:
            self._init_status_template()
        self._update_interval = update_interval
        self._recent_timing = []

        self._total_bar = FloatProgress(description='total',
                                        min=0, max=1, value=0,
                                        bar_style='info')
        self._total_html = HTML()
        self._epoch_bar = FloatProgress(description='this epoch',
                                        min=0, max=1, value=0,
                                        bar_style='info')
        self._epoch_html = HTML()
        self._status_html = HTML()

        self._widget = VBox([HBox([self._total_bar, self._total_html]),
                             HBox([self._epoch_bar, self._epoch_html]),
                             self._status_html])

    def initialize(self, manager):
        if self._training_length is None:
            t = manager._stop_trigger
            if not isinstance(t, trigger.IntervalTrigger):
                raise TypeError(
                    'cannot retrieve the training length from %s' % type(t))
            self._training_length = t.period, t.unit
            self._init_status_template()

        self.update(manager.iteration, manager.epoch_detail)
        display(self._widget)

    def __call__(self, manager):
        length, unit = self._training_length

        iteration, epoch_detail = manager.iteration, manager.epoch_detail

        if unit == 'iteration':
            is_finished = iteration == length
        else:
            is_finished = epoch_detail == length

        if iteration % self._update_interval == 0 or is_finished:
            self.update(iteration, epoch_detail)

    def finalize(self):
        if self._total_bar.value != 1:
            self._total_bar.bar_style = 'warning'
            self._epoch_bar.bar_style = 'warning'

    @property
    def widget(self):
        return self._widget

    def update(self, iteration, epoch_detail):
        length, unit = self._training_length

        recent_timing = self._recent_timing
        now = time.time()

        recent_timing.append((iteration, epoch_detail, now))

        if unit == 'iteration':
            rate = iteration / length
        else:
            rate = epoch_detail / length
        self._total_bar.value = rate
        self._total_html.value = "{:6.2%}".format(rate)

        epoch_rate = epoch_detail - int(epoch_detail)
        self._epoch_bar.value = epoch_rate
        self._epoch_html.value = "{:6.2%}".format(epoch_rate)

        status = self._status_template.format(iteration=iteration,
                                              epoch=int(epoch_detail))

        if rate == 1:
            self._total_bar.bar_style = 'success'
            self._epoch_bar.bar_style = 'success'

        old_t, old_e, old_sec = recent_timing[0]
        span = now - old_sec
        if span != 0:
            speed_t = (iteration - old_t) / span
            speed_e = (epoch_detail - old_e) / span
        else:
            speed_t = float('inf')
            speed_e = float('inf')

        if unit == 'iteration':
            estimated_time = (length - iteration) / speed_t
        else:
            estimated_time = (length - epoch_detail) / speed_e
        estimate = ('{:10.5g} iters/sec. Estimated time to finish: {}.'
                    .format(speed_t,
                            datetime.timedelta(seconds=estimated_time)))

        self._status_html.value = status + estimate

        if len(recent_timing) > 100:
            del recent_timing[0]

    def _init_status_template(self):
        self._status_template = (
            '{iteration:10} iter, {epoch} epoch / %s %ss<br />' %
            self._training_length)
