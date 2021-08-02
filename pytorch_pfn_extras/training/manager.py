import collections
import contextlib
import copy
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Union
from typing import TYPE_CHECKING
import warnings

import torch

from pytorch_pfn_extras import writing
from pytorch_pfn_extras import reporting
from pytorch_pfn_extras.training import extension as extension_module
from pytorch_pfn_extras.training import trigger as trigger_module
from pytorch_pfn_extras.training import _util as util_module

_get_time = time.perf_counter


class _ExtensionEntry:

    def __init__(
            self,
            extension: extension_module.Extension,
            priority: int,
            trigger: trigger_module.Trigger,
            call_before_training: bool
    ) -> None:
        self.extension = extension
        self.trigger = trigger
        self.priority = priority
        self.call_before_training = call_before_training

    def state_dict(self) -> Dict[str, Any]:
        state = {}
        state['extension'] = self.extension.state_dict()
        state['trigger'] = self.trigger.state_dict()
        return state

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        if 'extension' in to_load:
            self.extension.load_state_dict(to_load['extension'])
        if 'trigger' in to_load:
            self.trigger.load_state_dict(to_load['trigger'])


class _BaseExtensionsManager:
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
            stop_trigger: 'trigger_module.TriggerLike' = None
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
        self.observation: Dict[str, reporting.ReportValue] = {}
        self._out = out_dir
        self.writer = writer
        self.reporter = reporting.Reporter()
        self._start_extensions_called = False

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
            self.reporter.add_observer(name, model)
            self.reporter.add_observers(
                name, model.named_modules())
        self.max_epochs = max_epochs
        self._start_iteration = 0
        # Defer!
        self._start_time: Optional[float] = None
        self._iters_per_epoch: Optional[int] = None
        self._extensions: Dict[str, _ExtensionEntry] = collections.OrderedDict()
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

    @property
    def models(self) -> Dict[str, torch.nn.Module]:
        self.start_extensions()
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
        return self._stop_trigger(self)

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

    def _prepare_for_training(
            self,
            start_iteration: int,
            iters_per_epoch: int
    ) -> None:
        self.iteration = start_iteration
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
            extension: 'extension_module.ExtensionLike',
            name: Optional[str] = None,
            trigger: 'trigger_module.TriggerLike' = None,
            priority: Optional[int] = None,
            *,
            call_before_training: bool = False,
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
        ext = extension_module._as_extension(extension)
        if name is None:
            name = ext.name or ext.default_name
        if name == 'training':
            raise ValueError(
                'the name "training" is prohibited as an extension name')

        if trigger is None:
            trigger = ext.trigger
        trigger = trigger_module.get_trigger(trigger)

        if priority is None:
            priority = ext.priority

        modified_name = name
        ordinal = 0
        while modified_name in self._extensions:
            ordinal += 1
            modified_name = '%s_%d' % (name, ordinal)

        ext.name = modified_name
        self._extensions[modified_name] = _ExtensionEntry(
            ext, priority, trigger, call_before_training)

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

    def run_extensions(self) -> None:
        to_run = []
        for _, entry in self.extensions:
            if entry.trigger(self):
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
                    to_run.append(entry.extension)
                else:
                    entry.extension(self)
        for extension in to_run:
            extension(self)

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
            *,
            transform_models: Callable[
                [str, torch.nn.Module], torch.nn.Module] = lambda n, x: x
    ) -> Dict[str, Any]:
        """
        transform_models is a function that apply a transformation
        to a model.

        When using a `torch.nn.DataParallel` model, if we want
        to save only the `.module` object, state_dict can be
        called as follows:

        >>> manager.state_dict(transform_models=lambda n, x: x.module)
        """
        to_save: Dict[str, Any] = {}
        to_save['_start_iteration'] = self.iteration
        to_save['models'] = {
            name: transform_models(name, self._models[name]).state_dict()
            for name in self._models}
        to_save['optimizers'] = {name: self._optimizers[name].state_dict()
                                 for name in self._optimizers}
        to_save['extensions'] = {name: self._extensions[name].state_dict()
                                 for name in self._extensions}
        return to_save

    def load_state_dict(
            self,
            to_load: Dict[str, Any],
            *,
            transform_models: Callable[
                [str, torch.nn.Module], torch.nn.Module] = lambda n, x: x
    ) -> None:
        """
        transform_models is a function that apply a transformation
        to a model before loading its state.

        When using a `torch.nn.DataParallel` model, if we want
        to load the original state in a model with the
        `torch.nn.DataParallel` applied:

        >>> manager.load_state_dict(
                state, transform_models=(
                    lambda n, x: x.module
                    if isinstance(x, torch.nn.DataParallel) else x))
        """
        self._start_iteration = to_load['_start_iteration']
        self.iteration = self._start_iteration
        for name in self._models:
            # TODO(ecastill) map_loc when loading the model and DDP check
            transformed_model = transform_models(name, self._models[name])
            transformed_model.load_state_dict(to_load['models'][name])

        for name in self._optimizers:
            self._optimizers[name].load_state_dict(to_load['optimizers'][name])

        for name in self._extensions:
            self._extensions[name].load_state_dict(
                to_load['extensions'][name])


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
            writer: Optional[writing.Writer] = None
    ) -> None:
        super().__init__(
            models, optimizers, max_epochs, extensions,
            out_dir, writer, stop_trigger)
        if not (isinstance(iters_per_epoch, int) and iters_per_epoch >= 1):
            raise ValueError(
                'iters_per_epoch must be an integer >= 1 ({} given)'.format(
                    iters_per_epoch))
        self._prepare_for_training(0, iters_per_epoch)

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
        step_optimizers_names = []
        if step_optimizers is not None:
            step_optimizers_names = step_optimizers
        self.observation = {}
        with self.reporter.scope(self.observation):
            try:
                for name in step_optimizers_names:
                    self._optimizers[name].zero_grad()
                yield
                for name in step_optimizers_names:
                    self._optimizers[name].step()
            finally:
                # The iteration count is increased just before calling the
                # extensions.
                self.iteration += 1
                self.run_extensions()

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
            self._prepare_for_training(0, iters_per_epoch)
            self.start_extensions()
            start_iteration = self._start_iteration
            self.engine.state.iteration = self._start_iteration
            self.engine.state.epoch = self._start_epoch
            self._start_time = _get_time()
            # Initialize manager again after all state is restored
            self._prepare_for_training(start_iteration, iters_per_epoch)

            # Make all the next
            # handlers to be executed after user defined ones
            @self.engine.on(Events.ITERATION_COMPLETED)
            def run_extensions_on_iter(engine: Engine) -> None:
                self.iteration = engine.state.iteration
                self.run_extensions()

            # This should be the last extension to be run
            @self.engine.on(Events.ITERATION_COMPLETED)
            def close_reporter_on_iter(engine: Engine) -> None:
                self.cm.__exit__(None, None, None)

        @self.engine.on(Events.COMPLETED)
        def set_extensions_cleanup(engine: Engine) -> None:
            self._finalize_extensions()

    def state_dict(
            self,
            *,
            transform_models: Callable[
                [str, torch.nn.Module], torch.nn.Module] = lambda n, x: x
    ) -> Dict[str, Any]:
        to_save = super().state_dict(transform_models=transform_models)
        to_save['_epoch_length'] = self.engine.state.epoch_length
        to_save['_start_iteration'] = self.engine.state.iteration
        return to_save

    def load_state_dict(
            self,
            to_load: Dict[str, Any],
            *,
            transform_models: Callable[
                [str, torch.nn.Module], torch.nn.Module] = lambda n, x: x
    ) -> None:
        super().load_state_dict(to_load, transform_models=transform_models)
        self._start_epoch = self._start_iteration // to_load['_epoch_length']
