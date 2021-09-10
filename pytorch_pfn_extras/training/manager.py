# mypy: ignore-errors

import collections
import contextlib
import copy
from pytorch_pfn_extras.profiler import record
import time
from typing import Any, Dict, Generator, List, Optional, Union
from typing import TYPE_CHECKING
import warnings

import torch

from pytorch_pfn_extras import writing
from pytorch_pfn_extras import reporting
from pytorch_pfn_extras.training import extension as extension_module
from pytorch_pfn_extras.training import trigger as trigger_module
from pytorch_pfn_extras.training import _util as util_module
from pytorch_pfn_extras.training import _trigger_util
from pytorch_pfn_extras.training._transform_model import default_transform_model

_get_time = time.perf_counter


class IterationNotification:
    def __init__(self):
        self._is_completed = True

    def defer(self):
        self._is_completed = False


class _ManagerExecutionProxy(_trigger_util.TriggerArgument):
    """
    Object that is passed to the triggers of extensions depending if they
    require to measure executions when using async mode
    Note that in sync mode execution == iteration
    """
    def __init__(self, manager):
        self._manager = manager

    @property
    def iteration(self):
        return self._manager.execution

    @property
    def epoch(self):
        # Extensions will start via self.iteration
        return self.iteration // self._iters_per_epoch

    @property
    def epoch_detail(self):
        # Extensions will start via self.iteration
        return self.iteration / self._iters_per_epoch

    @property
    def _iters_per_epoch(self):
        return self._manager._iters_per_epoch


class _BaseExtensionsManager(_trigger_util.TriggerArgument):
    """
    Keeps track of the extensions and the current status
    """

    def __init__(
            self,
            models: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
            optimizers: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
            max_epochs: int,
            extensions: Optional[List['extension_module.ExtensionLike']],
            out_dir: str,
            writer: Optional[writing.Writer],
            stop_trigger: 'trigger_module.TriggerLike' = None,
            transform_model=default_transform_model,
    ) -> None:
        if extensions is None:
            extensions = []
        if stop_trigger is None:
            self._stop_trigger = trigger_module.get_trigger(
                (max_epochs, 'epoch'))
        else:
            self._stop_trigger = trigger_module.get_trigger(
                stop_trigger)
        if writer is None:
            writer = writing.SimpleWriter(out_dir=out_dir)
        # triggers are stateful, so we need to make a copy for internal use
        self._internal_stop_trigger = copy.deepcopy(self._stop_trigger)
        self.observation: reporting.Observation = {}
        self._out = out_dir
        self.writer = writer
        self.reporter = reporting.Reporter()
        self._transform_model = transform_model
        self._start_extensions_called = False

        # Indicates whether models can be accessed from extensions in the
        # current iteration.
        # The defualt value (True) indicates that it is allowed to access
        # models before starting a training loop.
        self._model_available = True

        if not isinstance(models, dict):
            if not isinstance(models, torch.nn.Module):
                raise ValueError(
                    'model must be an instance of dict or toch.nn.Module')
            self._models = {'main': models}
        else:
            self._models = models
        if not isinstance(optimizers, dict):
            # TODO(ecastill) Optimizer type is not checked because of tests
            # using mocks and other classes
            self._optimizers = {'main': optimizers}
        else:
            self._optimizers = optimizers

        for name, model in self._models.items():
            # TODO we should not initialize extensions at this point
            # so, we cannot use `self.models`
            model = self._transform_model(name, model)
            self.reporter.add_observer(name, model)
            self.reporter.add_observers(
                name, model.named_modules())
        self.max_epochs = max_epochs
        self._start_iteration = 0
        # Defer!
        self._start_time: Optional[float] = None
        self._iters_per_epoch: Optional[int] = None
        self._extensions: Dict[
            str, extension_module.ExtensionEntry] = collections.OrderedDict()
        for ext in extensions:
            self.extend(ext)

        # Initialize the writer
        self.writer.initialize(self.out)

    # All properties cannot be accessed without starting extensions, because
    # the snapshot extension must run first to restore the training state.

    @property
    def iteration(self) -> int:
        self.start_extensions()
        return self._iteration

    @iteration.setter
    def iteration(self, value: int) -> None:
        self._iteration = value

    def _check_model_available(self):
        if self._model_available:
            return
        raise RuntimeError(
            'Models cannot be accessed from extensions in this iteration. '
            'Extensions accessing models must declare '
            '`needs_model_state = True` attribute.')

    @property
    def models(self) -> Dict[str, torch.nn.Module]:
        self.start_extensions()
        self._check_model_available()
        models = {k: self._transform_model(k, v)
                  for k, v in self._models.items()}
        return models

    @property
    def raw_models(self):
        self.start_extensions()
        self._check_model_available()
        return self._models

    @property
    def optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        self.start_extensions()
        return self._optimizers

    @property
    def elapsed_time(self) -> float:
        if self._start_time is None:
            raise RuntimeError(
                'Unavailable until the initial run_iteration call.')
        return _get_time() - self._start_time

    @property
    def is_before_training(self) -> bool:
        # Extensions will start via self.iteration
        return self.iteration == 0

    @property
    def epoch(self) -> int:
        # Extensions will start via self.iteration
        assert self._iters_per_epoch is not None
        return self.iteration // self._iters_per_epoch

    @property
    def epoch_detail(self) -> float:
        # Extensions will start via self.iteration
        assert self._iters_per_epoch is not None
        return self.iteration / self._iters_per_epoch

    @property
    def stop_trigger(self) -> bool:
        self.start_extensions()
        # Trigger is stateful, we close the extensions the first time
        # it evaluates to True, as it won't do it again
        manager = self._get_proxy_for_trigger(self._stop_trigger)
        return self._stop_trigger(manager)

    @property
    def out(self) -> str:
        if self.writer.out_dir is not None:
            return self.writer.out_dir
        else:
            return self._out

    @property
    def updater(self) -> '_BaseExtensionsManager':
        warnings.warn(
            'The `updater` attribute has been deprecated in v0.3.0.'
            ' Use `iteration`, `epoch`, and `epoch_detail` attributes in'
            ' `ExtensionsManager` instead of attributes under `updater`.'
            ' You may also need to update the filename template specified to'
            ' snapshot extensions (e.g., from '
            '`snapshot_iter_{.updater.iteration}` to'
            ' `snapshot_iter_{.iteration}`).', DeprecationWarning)
        return self

    def _get_proxy_for_trigger(self, trigger):
        if isinstance(trigger, trigger_module.IntervalTrigger):
            return _ManagerExecutionProxy(self)
        return self

    def _prepare_for_training(
            self,
            start_iteration: int,
            start_execution: int,
            iters_per_epoch: int
    ) -> None:
        self.iteration = start_iteration
        self.execution = start_execution
        self._iters_per_epoch = iters_per_epoch

    def start_extensions(self) -> None:
        if self._start_extensions_called:
            # Extensions are already started or during the initialization.
            return
        else:
            self._start_extensions_called = True

        exts = self._extensions
        extension_order = sorted(
            exts.keys(),
            key=lambda name: exts[name].priority, reverse=True)
        self.extensions = [(name, exts[name])
                           for name in extension_order]

        # invoke initializer of each extension
        for _, entry in self.extensions:
            initializer = entry.extension.initialize
            finished = getattr(entry.trigger, 'finished', False)
            if not finished:
                initializer(self)

        # call extensions before training loop
        self.observation = {}
        with self.reporter.scope(self.observation):
            for _, entry in self.extensions:
                if entry.call_before_training:
                    entry.extension(self)

    def extend(
            self,
            extension: Union[
                'extension_module.ExtensionLike',
                'extension_module.ExtensionEntry',
            ],
            name: Optional[str] = None,
            trigger: 'trigger_module.TriggerLike' = None,
            priority: Optional[int] = None,
            *,
            call_before_training: Optional[bool] = None,
            **kwargs: Dict[str, Any],
    ) -> None:
        """Registers an extension to the manager.

        :class:`Extension` is a callable object which is called after each
        update unless the corresponding trigger object decides to skip the
        iteration. The order of execution is determined by priorities:
        extensions with higher priorities are called earlier in each iteration.
        Extensions with the same priority are invoked in the order of
        registrations.

        If two or more extensions with the same name are registered, suffixes
        are added to the names of the second to last extensions. The suffix is
        ``_N`` where N is the ordinal of the extensions.

        See :class:`Extension` for the interface of extensions.

        Args:
            extension: Extension to register.
            name (str): Name of the extension. If it is omitted, the
                :attr:`Extension.name` attribute of the extension is used or
                the :attr:`Extension.default_name` attribute of the extension
                if `name` is is set to `None` or is undefined.
                Note that the name would be suffixed by an ordinal in case of
                duplicated names as explained above.
            trigger (tuple or Trigger): Trigger object that determines when to
                invoke the extension. If it is ``None``, ``extension.trigger``
                is used instead. If it is ``None`` and the extension does not
                have the trigger attribute, the extension is triggered at every
                iteration by default. If the trigger is not callable, it is
                passed to :class:`IntervalTrigger` to build an interval
                trigger.
            call_before_training (bool): Flag to call extension before
                training. Default is ``False``.
            priority (int): Invocation priority of the extension. Extensions
                are invoked in the descending order of priorities in each
                iteration. If this is ``None``, ``extension.priority`` is used
                instead.

        """
        if self._start_extensions_called:
            raise RuntimeError(
                'extend called after the extensions were initialized')

        if isinstance(extension, extension_module.ExtensionEntry):
            entry = extension
        else:
            entry = extension_module.ExtensionEntry(extension)

        if trigger is not None:
            entry._update_trigger(trigger)

        if priority is not None:
            entry.priority = priority

        if call_before_training is not None:
            entry.call_before_training = call_before_training

        modified_name = name or entry.name
        ordinal = 0
        while modified_name in self._extensions:
            ordinal += 1
            modified_name = '%s_%d' % (name, ordinal)

        entry._update_name(modified_name)
        self._extensions[modified_name] = entry

    def get_extension(self, name: str) -> extension_module.Extension:
        """Returns the extension of a given name.

        Args:
            name (str): Name of the extension.

        Returns:
            Extension.

        """
        extensions = self._extensions
        if name in extensions:
            return extensions[name].extension
        else:
            raise ValueError('extension %s not found' % name)

    def run_extensions(self, *, completed=True, only_iterations=True) -> None:
        if completed:
            # Check if the model is available for the iteration just
            # completed, i.e., the iteration number is already incremented.
            self._model_available = self.needs_model_state(self.iteration)
        else:
            self._model_available = False

        to_run = []
        for name, entry in self.extensions:
            # When iterations are deferred we only
            # launch the extensions that doesn't need
            # the training status to advance
            # those are extensions set to execute
            # in a given interval of executions
            is_async = (hasattr(entry.extension, 'is_async')
                        and entry.extension.is_async)
            if ((not completed and not is_async)
                    or (completed and is_async and only_iterations)):
                continue
            manager = self
            if is_async:
                manager = self._get_proxy_for_trigger(entry.trigger)
            if entry.trigger(manager):
                # Execution of snapshot extensions are deferred until all the
                # triggers are evaluated.
                # If we don't do this, when two (or more) snapshot extensions
                # are registered and triggers for them are stateful, the first
                # snapshot extension will save the state of the second trigger
                # before invoking it although it will be executed later in this
                # iteration, making them to fire again just after resuming from
                # the snaphsot saved by the first snapshot extension.
                # Non-snapshot extensions are executed right away (note that
                # the order is already sorted by the priority) as they will
                # report values that might be needed by other triggers, i.e.,
                # trigger based on evaluator reported value.
                if entry.priority == extension_module.PRIORITY_SNAPSHOT:
                    to_run.append((name, entry.extension))
                else:
                    with record(
                        f'pytorch_pfn_extras.training.ExtensionsManager'
                        f'.run_extensions:{name}'
                    ):
                        entry.extension(self)
        for name, extension in to_run:
            with record(
                f'pytorch_pfn_extras.training.ExtensionsManager'
                f'.run_extensions:{name}'
            ):
                extension(self)
        self._model_available = True

    def needs_state_this_iteration(self):
        # TODO(kmaehashi) remove this interface after migration complete.
        return self.needs_model_state(self.execution + 1)

    def needs_model_state(self, iteration=None):
        if iteration is None:
            # Iteration is added one, because iteration count
            # is increased just right before calling extensions
            iteration = self.iteration + 1
        for _, entry in self._extensions.items():
            needs_state = getattr(entry.extension, 'needs_model_state', False)
            if (needs_state and entry.trigger.may_fire(
                    iteration, self._iters_per_epoch)):
                return True
        return False

    def _finalize_extensions(self) -> None:
        for _, entry in self.extensions:
            # Some mock objects for tests give errors
            # if we use `getattr`
            try:
                if entry.extension.finalize:
                    entry.extension.finalize()
            except AttributeError:
                pass

    def state_dict(
            self,
    ) -> Dict[str, Any]:
        to_save: Dict[str, Any] = {}
        to_save['_start_iteration'] = self.iteration
        to_save['_start_execution'] = self.execution
        # Use self.models to apply transform_model
        to_save['models'] = {
            name: self.models[name].state_dict()
            for name in self.models}
        to_save['optimizers'] = {name: self._optimizers[name].state_dict()
                                 for name in self._optimizers}
        to_save['extensions'] = {name: self._extensions[name].state_dict()
                                 for name in self._extensions}
        return to_save

    def load_state_dict(
            self,
            to_load: Dict[str, Any],
    ) -> None:
        self._start_iteration = to_load['_start_iteration']
        self.iteration = self._start_iteration
        self._start_execution = to_load.get('_start_execution', self.iteration)
        self.execution = self._start_execution
        for name in self.models:
            # TODO(ecastill) map_loc when loading the model and DDP check
            # Use self.models to apply transform_model
            self.models[name].load_state_dict(to_load['models'][name])

        for name in self._optimizers:
            self._optimizers[name].load_state_dict(to_load['optimizers'][name])

        for name in self._extensions:
            self._extensions[name].load_state_dict(to_load['extensions'][name])


class ExtensionsManager(_BaseExtensionsManager):
    """Manages the extensions and the current status.

    Args:
        models (dict or `torch.nn.Module`): Map of string to Module
            or an actual Module
        optimizers (dict or `torch.Optimizer`): Map of string to Optimizer
            or an actual Optimizer.
        max_epochs (int): Number of epochs in the whole training loop. Ignored
            if `stop_trigger` is passed as a kwarg.
        iters_per_epoch (int): Number of iterations in one epoch.
        extensions (list or None): List of Extentions to be used.
        out_dir (str): Output directory (default: ``result``).
        stop_trigger (trigger object, optional) trigger that can be consulted
           to determine wether training has concluded. The default is an
           interval trigger set to `max_epochs`
        writer (writing.Writer object): Writer that can be used by
            extensions to write data to custom filesystems.
    """

    def __init__(
            self,
            models: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
            optimizers: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
            max_epochs: int,
            *,
            iters_per_epoch: Optional[int],
            extensions: Optional[List['extension_module.ExtensionLike']] = None,
            out_dir: str = 'result',
            stop_trigger: 'trigger_module.TriggerLike' = None,
            writer: Optional[writing.Writer] = None,
            transform_model=lambda n, x: x,
    ) -> None:
        super().__init__(
            models, optimizers, max_epochs, extensions,
            out_dir, writer, stop_trigger, transform_model)
        if not (isinstance(iters_per_epoch, int) and iters_per_epoch >= 1):
            raise ValueError(
                'iters_per_epoch must be an integer >= 1 ({} given)'.format(
                    iters_per_epoch))
        self._prepare_for_training(0, 0, iters_per_epoch)

    @contextlib.contextmanager
    def complete_iteration(self, *, observation=None, step_optimizers=None):
        """ Context manager to complete deferred iterations.

        Args:
            step_optimizers (list or None): names of the optimizers
            to call `zero_grad` and `step`
        """

        # This manager is needed because we need a reporting environment
        # for extensions to work properly.

        if observation is None:
            observation = {}
        step_optimizers_names = []
        if step_optimizers is not None:
            step_optimizers_names = step_optimizers
        self.observation = observation
        with self.reporter.scope(observation):
            try:
                for name in step_optimizers_names:
                    self._optimizers[name].zero_grad()
                yield
                for name in step_optimizers_names:
                    self._optimizers[name].step()
            finally:
                self.iteration += 1
                self.run_extensions(completed=True, only_iterations=True)

    @contextlib.contextmanager
    def run_iteration(
            self,
            *,
            step_optimizers: Optional[List[str]] = None
    ) -> Generator[None, None, None]:
        """Context manager to run an iteration.

        This manager can additionally run a step in the
        specified optimizers names.

        Args:
            step_optimizers (list or None): names of the optimizers
            to call `zero_grad` and `step`
        """
        if self._start_time is None:
            self._start_time = _get_time()
            self.start_extensions()

        notification = IterationNotification()
        step_optimizers_names = []
        if step_optimizers is not None:
            step_optimizers_names = step_optimizers
        self.observation = {}
        with self.reporter.scope(self.observation):
            try:
                for name in step_optimizers_names:
                    self._optimizers[name].zero_grad()
                yield notification
                if notification._is_completed:
                    for name in step_optimizers_names:
                        self._optimizers[name].step()
            finally:
                # The iteration count is increased just before calling the
                # extensions.
                if notification._is_completed:
                    self.iteration += 1
                    self.execution += 1
                    self.run_extensions(completed=True, only_iterations=False)
                else:
                    # execution is the number of times
                    # a run_iteration event was not complete
                    # but operations on the model were performed.
                    # it is used to trigger extensions that
                    # needs to run regardless of the training state
                    self.execution += 1
                    self.run_extensions(completed=False, only_iterations=False)

        if self._internal_stop_trigger(self):
            self._finalize_extensions()


if TYPE_CHECKING:
    import ignite


class IgniteExtensionsManager(_BaseExtensionsManager):
    """Manages extensions and the current status in Ignite training loop.

    Args:
        engine (ignite.engine.Engine): Ignite trainer engine
        models (dict or torch.nn.Module): Map of string to Module
            or an actual Module
        optimizers (dict or torch.Optimizer): Map of string to Optimizer
            or an actual Optimizer.
        max_epochs (int): Number of epochs in the whole training loop.
        extensions (list or None): List of Extentions to be used.
        out_dir (str): Output directory (default: ``result``).
        writer (writing.Writer object): Writer that can be used by
            extensions to write data to custom filesystems.
    """
    def __init__(
            self,
            engine: 'ignite.engine.Engine',
            models: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
            optimizers: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
            max_epochs: int,
            *,
            extensions: Optional[List['extension_module.ExtensionLike']] = None,
            out_dir: str = 'result',
            writer: Optional[writing.Writer] = None
    ) -> None:
        import ignite
        if not isinstance(engine, ignite.engine.Engine):
            raise TypeError("Argument 'engine' must be of ignite.Engine type.")
        if (util_module._get_ignite_version(ignite.__version__)
                < util_module._get_ignite_version('0.3.0')):
            raise ImportError('Ignite version found {}. '
                              'Required is >=0.3.0'.format(ignite.__version__))
        super().__init__(
            models, optimizers, max_epochs, extensions, out_dir, writer)
        self.engine = engine
        self._start_epoch = 0  # Used to correctly restore snapshots
        self.set_ignite_handlers()

    def set_ignite_handlers(self) -> None:
        from ignite.engine import Engine
        from ignite.engine import Events

        # Set a handler that sets the reporter scope on every iteration
        @self.engine.on(Events.ITERATION_STARTED)
        def set_reporter_on_iter(engine: Engine) -> None:
            self.observation = {}
            self.cm = self.reporter.scope(self.observation)
            self.cm.__enter__()

        @self.engine.on(Events.STARTED)
        def set_training_started(engine: Engine) -> None:
            iters_per_epoch = len(engine.state.dataloader)
            # Initialize manager once before extensions' `initialize` call
            self._prepare_for_training(0, 0, iters_per_epoch)
            self.start_extensions()
            start_iteration = self._start_iteration
            self.engine.state.iteration = self._start_iteration
            self.engine.state.epoch = self._start_epoch
            self._start_time = _get_time()
            # Initialize manager again after all state is restored
            self._prepare_for_training(
                start_iteration, start_iteration, iters_per_epoch)

            # Make all the next
            # handlers to be executed after user defined ones
            @self.engine.on(Events.ITERATION_COMPLETED)
            def run_extensions_on_iter(engine: Engine) -> None:
                self.iteration = engine.state.iteration
                self.execution += 1
                self.run_extensions()

            # This should be the last extension to be run
            @self.engine.on(Events.ITERATION_COMPLETED)
            def close_reporter_on_iter(engine: Engine) -> None:
                self.cm.__exit__(None, None, None)

        @self.engine.on(Events.COMPLETED)
        def set_extensions_cleanup(engine: Engine) -> None:
            self._finalize_extensions()

    def state_dict(self) -> Dict[str, Any]:
        to_save = super().state_dict()
        to_save['_epoch_length'] = self.engine.state.epoch_length
        to_save['_start_iteration'] = self.engine.state.iteration
        return to_save

    def load_state_dict(
            self,
            to_load: Dict[str, Any],
    ) -> None:
        super().load_state_dict(to_load)
        self._start_epoch = self._start_iteration // to_load['_epoch_length']
