import datetime

import six

import torch
import pytorch_extensions.convert as convert
from pytorch_extensions import reporter as reporter_module
from pytorch_extensions import extension
from pytorch_extensions.extensions import util


class Evaluator(extension.Extension):

    """__init__(self, iterator, target, eval_func=None, *, progress_bar=False)

    Trainer extension to evaluate models on a validation set.

    This extension evaluates the current models by a given evaluation function.
    It creates a :class:`~Reporter` object to store values observed in
    the evaluation function on each iteration. The report for all iterations
    are aggregated to :class:`~DictSummary`. The collected mean values
    are further reported to the reporter object of the trainer, where the name
    of each observation is prefixed by the evaluator name. See
    :class:`~Reporter` for details in naming rules of the reports.

    Evaluator has a structure to customize similar to that of
    :class:`~StandardUpdater`.
    The main differences are:

    - There are no optimizers in an evaluator. Instead, it holds links
      to evaluate.
    - An evaluation loop function is used instead of an update function.
    - Preparation routine can be customized, which is called before each
      evaluation. It can be used, e.g., to initialize the state of stateful
      recurrent networks.

    There are two ways to modify the evaluation behavior besides setting a
    custom evaluation function. One is by setting a custom evaluation loop via
    the ``eval_func`` argument. The other is by inheriting this class and
    overriding the :meth:`evaluate` method. In latter case, users have to
    create and handle a reporter object manually. Users also have to copy the
    iterators before using them, in order to reuse them at the next time of
    evaluation. In both cases, the functions are called in testing mode

    This extension is called at the end of each epoch by default.

    Args:
        iterator: Dataset iterator for the validation dataset. It can also be
            a dictionary of iterators. If this is just an iterator, the
            iterator is registered by the name ``'main'``.
        target: Link object or a dictionary of links to evaluate. If this is
            just a link object, the link is registered by the name ``'main'``.
        eval_func: Evaluation function called at each iteration. The target
            link to evaluate as a callable is used by default.
        progress_bar: Boolean flag to show a progress bar while training,
            which is similar to
            :class:`~extensions.ProgressBar`.
            (default: ``False``)

    .. warning::

        The argument ``progress_bar`` is experimental.
        The interface can change in the future.

    Attributes:
        converter: Converter function.
        device: Device to which the validation data is sent.
        eval_hook: Function to prepare for each evaluation process.
        eval_func: Evaluation function called at each iteration.

    """
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    name = None

    def __init__(self, iterator, target, converter=convert.transfer_data,
                 device=None, eval_hook=None, eval_func=None, **kwargs):
        progress_bar = kwargs.get('progress_bar', False)

        if isinstance(iterator, torch.utils.data.DataLoader):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(target, torch.nn.Module):
            self._targets = {'main': target}

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.eval_func = eval_func
        self._progress_bar = progress_bar

    def get_iterator(self, name):
        """Returns the iterator of the given name."""
        return self._iterators[name]

    def get_all_iterators(self):
        """Returns a dictionary of all iterators."""
        return dict(self._iterators)

    def get_target(self, name):
        """Returns the target link of the given name."""
        return self._targets[name]

    def get_all_targets(self):
        """Returns a dictionary of all target links."""
        return dict(self._targets)

    def __call__(self, trainer=None):
        """Executes the evaluator extension.

        Unlike usual extensions, this extension can be executed without passing
        a trainer object. This extension reports the performance on validation
        dataset using the :func:`~reporter_module.report` function.
        Thus, users can use this extension independently from any trainer
        by manually configuring a :class:`~Reporter` object.

        Args:
            trainer (~ExtensionsManager): Trainer object that invokes
                this extension. It can be omitted in case of calling this
                extension manually.

        Returns:
            dict: Result dictionary that contains mean statistics of values
            reported by the evaluation function.

        """
        # set up a reporter
        reporter = reporter_module.Reporter()
        if self.name is not None:
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.named_modules())

        with reporter:
            with torch.no_grad():
                result = self.evaluate()

        reporter_module.report(result)
        return result

    def evaluate(self):
        """Evaluates the model and returns a result dictionary.

        This method runs the evaluation loop over the validation dataset. It
        accumulates the reported values to :class:`~DictSummary` and
        returns a dictionary whose values are means computed by the summary.

        Users can override this method to customize the evaluation routine.

        Returns:
            dict: Result dictionary. This dictionary is further reported via
            :func:`~pytorch_extensions.report` without specifying any observer.

        """
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        summary = reporter_module.DictSummary()

        updater = IterationStatus(len(iterator))
        if self._progress_bar:
            pbar = _IteratorProgressBar(iterator=updater)

        for idx, batch in enumerate(iterator):
            updater.current_position = idx
            in_arrays = convert._call_converter(
                    self.converter, batch, self.device)
            observation = {}
            with reporter_module.report_scope(observation):
                eval_func(*in_arrays)
            summary.add(observation)

            if self._progress_bar:
                pbar.update()
        if self._progress_bar:
            pbar.close()

        return summary.compute_mean()

    def finalize(self):
        """Finalizes the evaluator object.

        This method calls the `finalize` method of each iterator that
        this evaluator has.
        It is called at the end of training loops.

        """
        # for iterator in six.itervalues(self._iterators):
        #     iterator.finalize()
        pass


class IterationStatus(object):
    def __init__(self, size):
        self.current_position = 0
        self._epoch_detail = 0.0
        self._size = size

    @property
    def epoch_detail(self):
        return self.current_position/self._size


class _IteratorProgressBar(util.ProgressBar):

    def __init__(self, iterator, bar_length=None, out=None):
        if not (hasattr(iterator, 'current_position') and
                hasattr(iterator, 'epoch_detail')):
            raise TypeError('Iterator must have the following attributes '
                            'to enable a progress bar: '
                            'current_position, epoch_detail')
        self._iterator = iterator

        super(_IteratorProgressBar, self).__init__(
            bar_length=bar_length, out=out)

    def get_lines(self):
        iteration = self._iterator.current_position
        epoch_detail = self._iterator.epoch_detail
        epoch_size = getattr(self._iterator, '_epoch_size', None)

        lines = []

        rate = epoch_detail
        marks = '#' * int(rate * self._bar_length)
        lines.append('validation [{}{}] {:6.2%}\n'.format(
                     marks, '.' * (self._bar_length - len(marks)), rate))

        if epoch_size:
            lines.append('{:10} / {} iterations\n'
                         .format(iteration, epoch_size))
        else:
            lines.append('{:10} iterations\n'.format(iteration))

        speed_t, speed_e = self.update_speed(iteration, epoch_detail)
        estimated_time = (1.0 - epoch_detail) / speed_e
        lines.append('{:10.5g} iters/sec. Estimated time to finish: {}.\n'
                     .format(speed_t,
                             datetime.timedelta(seconds=estimated_time)))
        return lines


class IgniteEvaluator(Evaluator):
    def __init__(self, evaluator, iterator, target, **kwargs):
        super().__init__(iterator, target, None, **kwargs)
        self.evaluator = evaluator
        self.set_evaluator_handlers()

    def set_evaluator_handlers(self):
        from ignite.engine import Events
        # Register handlers to retrieve the Average metrics and report them
        @self.evaluator.on(Events.EPOCH_STARTED)
        def set_evaluation_started(engine):
            self.observation = {}
            self.cm = reporter_module.report_scope(self.observation)
            self.cm.__enter__()

        if self._progress_bar:
            @self.evaluator.on(Events.ITERATION_STARTED)
            def update_progress_bar(engine):
                self.updater.current_position = engine.state.iteration
                self.pbar.update()

        @self.evaluator.on(Events.EPOCH_COMPLETED)
        def set_evaluation_completed(engine):
            metrics = self.evaluator.state.metrics
            for metric in metrics:
                reporter_module.report(
                    {'val/{}'.format(metric): metrics[metric]})
            self.cm.__exit__(None, None, None)
            self.summary.add(self.observation)

    def evaluate(self):
        iterator = self._iterators['main']
        self.summary = reporter_module.DictSummary()
        self.updater = IterationStatus(len(iterator))
        if self._progress_bar:
            self.pbar = _IteratorProgressBar(iterator=self.updater)
        self.evaluator.run(iterator)
        if self._progress_bar:
            self.pbar.close()
        return self.summary.compute_mean()
