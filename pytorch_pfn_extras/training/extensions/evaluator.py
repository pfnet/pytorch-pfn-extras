import contextlib
import datetime
from typing import (
    Any, Callable, Dict, Generator, Iterable, List, Optional, TextIO, Union,
    TYPE_CHECKING,
)

import numpy
import torch
import torch.distributed

from pytorch_pfn_extras import reporting
from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training.extensions import util
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol


_MetricType = Callable[[Any, Any, Any], None]
_Scalar = Union[torch.Tensor, numpy.ndarray, numpy.floating, float]


class Evaluator(extension.Extension):

    """__init__(self, iterator, target, eval_func=None, *, progress_bar=False)

    An extension to evaluate models on a validation set.

    This extension evaluates the current models by a given evaluation function.
    It creates a :class:`~Reporter` object to store values observed in
    the evaluation function on each iteration. The report for all iterations
    are aggregated to :class:`~DictSummary`. The collected mean values
    are further reported to the reporter object of the manager, where the name
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
        target: torch.nn.Module object or a dictionary of links to evaluate.
            If this is just a layer object, the link is registered by the
            name ``'main'``.
        eval_func: Evaluation function called at each iteration. The target
            link to evaluate as a callable is used by default.
        progress_bar: Boolean flag to show a progress bar while training,
            which is similar to
            :class:`~extensions.ProgressBar`.
            (default: ``False``)
        metrics: List of callables that are called every batch to
            calculate metrics such as accuracy, roc_auc or others
            The signature of the callable is:
            `def metric_fn(batch, output, last_iteration)`
            (default: ``[]``)

    .. warning::

        The argument ``progress_bar`` is experimental.
        The interface can change in the future.

    Attributes:
        eval_hook: Function to prepare for each evaluation process.
        eval_func: Evaluation function called at each iteration.

    """
    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    def __init__(
            self,
            iterator: Union[torch.utils.data.DataLoader[Any],
                            Dict[str, torch.utils.data.DataLoader[Any]]],
            target: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
            eval_hook: Optional[Callable[['Evaluator'], None]] = None,
            eval_func: Optional[Callable[..., Any]] = None,
            **kwargs: Any,
    ) -> None:
        progress_bar = kwargs.get('progress_bar', False)
        metrics = kwargs.get('metrics', [])

        if isinstance(iterator, torch.utils.data.DataLoader):
            self._iterators = {'main': iterator}
        else:
            self._iterators = iterator

        if isinstance(target, torch.nn.Module):
            target = {'main': target}
        self._targets = target

        self.name = None
        self.eval_hook = eval_hook
        self._eval_func = eval_func
        self._progress_bar = progress_bar
        self._metrics = metrics

    def eval_func(self, *args: Any, **kwargs: Any) -> Any:
        if self._eval_func:
            func = self._eval_func
        else:
            func = self._targets['main']
        return func(*args, **kwargs)

    def get_iterator(self, name: str) -> torch.utils.data.DataLoader[Any]:
        """Returns the iterator of the given name."""
        return self._iterators[name]

    def get_all_iterators(self) -> Dict[str, torch.utils.data.DataLoader[Any]]:
        """Returns a dictionary of all iterators."""
        return dict(self._iterators)

    def get_target(self, name: str) -> torch.nn.Module:
        """Returns the target link of the given name."""
        return self._targets[name]

    def get_all_targets(self) -> Dict[str, torch.nn.Module]:
        """Returns a dictionary of all target links."""
        return dict(self._targets)

    def add_metric(self, metric_fn: _MetricType) -> None:
        """Adds a custom metric to the evaluator.

        The metric is a callable that is executed every batch
        with the following signature:
        `def metric_fn(batch, output, last_iteration)`

        Batch is the input batch passed to the model. Output
        is the result of evaluating batch, last_iteration is
        a boolean flag that indicates if its the last batch
        in the evaluation.
        """
        self._metrics.append(metric_fn)

    def __call__(
            self,
            manager: Optional[ExtensionsManagerProtocol] = None,
    ) -> Optional[Dict[str, _Scalar]]:
        """Executes the evaluator extension.

        Unlike usual extensions, this extension can be executed without passing
        a manager object. This extension reports the performance on validation
        dataset using the :func:`~reporting.report` function.
        Thus, users can use this extension independently from any manager
        by manually configuring a :class:`~Reporter` object.

        Args:
            manager (~pytorch_pfn_extras.training.ExtensionsManager): Manager
                object that invokes this extension.

        Returns:
            dict: Result dictionary that contains mean statistics of values
            reported by the evaluation function.

        """
        # set up a reporter
        reporter = reporting.Reporter()
        if self.name is not None:
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in self._targets.items():
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.named_modules())

        with reporter:
            with torch.no_grad():  # type: ignore[no-untyped-call]
                result = self.evaluate()

        reporting.report(result)
        return result

    def _gather_summaries(self, summary: reporting.DictSummary) -> reporting.DictSummary:
        return summary

    def evaluate(self) -> Dict[str, _Scalar]:
        """Evaluates the model and returns a result dictionary.

        This method runs the evaluation loop over the validation dataset. It
        accumulates the reported values to :class:`~DictSummary` and
        returns a dictionary whose values are means computed by the summary.

        Users can override this method to customize the evaluation routine.

        Returns:
            dict: Result dictionary. This dictionary is further reported via
            :func:`~pytorch_pfn_extras.report` without specifying any observer.

        """
        iterator = self._iterators['main']

        if self.eval_hook:
            self.eval_hook(self)

        summary = reporting.DictSummary()

        progress = IterationStatus(len(iterator))
        if self._progress_bar:
            name = self.name or self.default_name
            pbar = _IteratorProgressBar(name=name, iterator=progress)

        last_iter = len(iterator) - 1
        with _in_eval_mode(self._targets.values()):
            for idx, batch in enumerate(iterator):
                last_batch = idx == last_iter
                progress.current_position = idx
                observation: Dict[str, Any] = {}
                with reporting.report_scope(observation):
                    if isinstance(batch, tuple) and hasattr(batch, '_fields'):
                        outs = self.eval_func(batch)
                    elif isinstance(batch, (tuple, list)):
                        outs = self.eval_func(*batch)
                    elif isinstance(batch, dict):
                        outs = self.eval_func(**batch)
                    else:
                        outs = self.eval_func(batch)
                    for metric in self._metrics:
                        metric(batch, outs, last_batch)
                summary.add(observation)

                if self._progress_bar:
                    pbar.update()

        if self._progress_bar:
            pbar.close()

        summary = self._gather_summaries(summary)

        return summary.compute_mean()


def _dist_gather(obj: Any) -> List[Any]:
    world_size = torch.distributed.get_world_size()  # type: ignore[no-untyped-call]
    placeholder = [object() for _ in range(world_size)]
    torch.distributed.all_gather_object(placeholder, obj)  # type: ignore[no-untyped-call]
    return placeholder


class DistributedEvaluator(Evaluator):

    """__init__(self, iterator, target, eval_func=None, *, progress_bar=False)

    An extension to evaluate models on a validation set in a distributed training setup.

    In case torch.distributed is used to parallelize training iterations,
    it is efficient to also run evaluation in parallel by splitting the validation set
    to each worker process and conduct evaluation separately followed by aggregation
    of results of each worker, which can be achieved by :class:~`DistributedEvaluator`.

    This extension basically behaves similarly to :class:`~Evaluator`,
    but adds an aggregation step in :func:`Evaluator.evaluate`.
    A summary of evaluation (:class:`~DictSummary`) in each worker process
    is collected in "all-gather" manner and then accumulated.
    Therefore all the worker processes must attend the evaluation,
    i.e., make sure all the processes have a :class:`~Evaluator` extension object
    configured in the :class:`~ExtensionManager` with the same trigger.
    All the worker process will get identical evaluation result returned by :func:`Evaluator.evaluate`
    and reported to an observation.

    It is necessary to pass a DataLoader with an appropripate sampler which properly
    splits the validation dataset to each MPI worker process.
    PyTorch DistributedSampler implements this, but it allows sampler repetition
    in order to make the number of samples assigned to each process identical.
    For evaluation purpose it distorts the evaluation result,
    hence it is recommended to use :class:`~DistributedValidationSampler` instead.

    """

    def __init__(
            self,
            iterator: Union[torch.utils.data.DataLoader[Any],
                            Dict[str, torch.utils.data.DataLoader[Any]]],
            target: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
            eval_hook: Optional[Callable[['Evaluator'], None]] = None,
            eval_func: Optional[Callable[..., Any]] = None,
            **kwargs: Any,
    ) -> None:
        if not torch.distributed.is_initialized():  # type: ignore[no-untyped-call]
            msg = "PyTorch distributed module is not initialized. " \
                  "Initialize process group or use non-distributed Evaluator."
            raise RuntimeError(msg)

        if 'progress_bar' in kwargs:
            rank = torch.distributed.get_rank()  # type: ignore[no-untyped-call]
            kwargs['progress_bar'] &= (rank == 0)

        super().__init__(iterator, target, eval_hook, eval_func, **kwargs)

    def _gather_summaries(self, summary: reporting.DictSummary) -> reporting.DictSummary:
        return sum(_dist_gather(summary), reporting.DictSummary())


@contextlib.contextmanager
def _in_eval_mode(targets: Iterable[torch.nn.Module]) -> Generator[None, None, None]:
    targets = list(targets)
    was_train = [t.training for t in targets]
    try:
        for t in targets:
            t.eval()
        yield
    finally:
        for t, was in zip(targets, was_train):
            t.train(was)


class IterationStatus:
    def __init__(self, size: int) -> None:
        self.current_position = 0
        self._epoch_detail = 0.0
        self._size = size

    @property
    def epoch_detail(self) -> float:
        return self.current_position / self._size


class _IteratorProgressBar(util.ProgressBar):

    def __init__(
            self,
            name: str,
            iterator: IterationStatus,
            bar_length: int = 50,
            out: Optional[TextIO] = None,
    ):
        if not (hasattr(iterator, 'current_position')
                and hasattr(iterator, 'epoch_detail')):
            raise TypeError('Iterator must have the following attributes '
                            'to enable a progress bar: '
                            'current_position, epoch_detail')
        self._name = name
        self._iterator = iterator
        self._bar_length = bar_length

        super().__init__(out=out)

    def get_lines(self) -> List[str]:
        iteration = self._iterator.current_position
        epoch_detail = self._iterator.epoch_detail
        epoch_size = getattr(self._iterator, '_epoch_size', None)

        lines = []

        rate = epoch_detail
        marks = '#' * int(rate * self._bar_length)
        rest_marks = '.' * (self._bar_length - len(marks))
        lines.append('{} [{}{}] {:6.2%}\n'.format(
                     self._name, marks, rest_marks, rate))

        if epoch_size:
            lines.append(f'{{:{len(self._name)}}} / {{}} iterations\n'
                         .format(iteration, epoch_size))
        else:
            lines.append(f'{{:{len(self._name)}}} iterations\n'
                         .format(iteration))

        speed_t, speed_e = self.update_speed(iteration, epoch_detail)
        estimated_time = (1.0 - epoch_detail) / speed_e
        itps = f'{{:{len(self._name)}.5g}} iters/sec.'.format(speed_t)
        eta = 'Estimated time to finish: {}.\n' \
              .format(datetime.timedelta(seconds=estimated_time))
        lines.append("{} {}".format(itps, eta))
        return lines


if TYPE_CHECKING:
    from typing.ignite import Engine


class IgniteEvaluator(Evaluator):
    def __init__(
            self,
            evaluator: 'Engine',
            iterator: Union[torch.utils.data.DataLoader[Any],
                            Dict[str, torch.utils.data.DataLoader[Any]]],
            target: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
            **kwargs: Any,
    ):
        super().__init__(iterator, target, None, **kwargs)
        self.evaluator = evaluator
        self.set_evaluator_handlers()

    def set_evaluator_handlers(self) -> None:
        from ignite.engine import Engine
        from ignite.engine import Events

        # Register handlers to retrieve the Average metrics and report them
        @self.evaluator.on(Events.ITERATION_STARTED)
        def set_evaluation_started(engine: Engine) -> None:
            self.observation: Dict[str, Any] = {}
            self.cm = reporting.report_scope(self.observation)
            self.cm.__enter__()

        if self._progress_bar:
            @self.evaluator.on(Events.ITERATION_STARTED)
            def update_progress_bar(engine: Engine) -> None:
                self.progress.current_position = engine.state.iteration
                self.pbar.update()

        @self.evaluator.on(Events.ITERATION_COMPLETED)
        def report_iteration_metrics(engine: Engine) -> None:
            self.summary.add(self.observation)
            self.cm.__exit__(None, None, None)

        @self.evaluator.on(Events.EPOCH_COMPLETED)
        def set_evaluation_completed(engine: Engine) -> None:
            ignite_metrics: Dict[str, Any] = {}
            with reporting.report_scope(ignite_metrics):
                metrics = self.evaluator.state.metrics
                for metric in metrics:
                    reporting.report(
                        {'val/{}'.format(metric): metrics[metric]})
                self.summary.add(ignite_metrics)

    def evaluate(self) -> Dict[str, _Scalar]:
        iterator = self._iterators['main']
        self.summary = reporting.DictSummary()
        self.progress = IterationStatus(len(iterator))
        if self._progress_bar:
            name = self.name or self.default_name
            self.pbar = _IteratorProgressBar(name=name, iterator=self.progress)
        self.evaluator.run(iterator)
        if self._progress_bar:
            self.pbar.close()
        return self.summary.compute_mean()
