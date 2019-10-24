import collections
import contextlib
import os
import time

from pytorch_extensions import extension as extension_module
from pytorch_extensions import trigger as trigger_module
from pytorch_extensions.reporter import Reporter


# Select the best-resolution timer function
try:
    _get_time = time.perf_counter
except AttributeError:
    if os.name == 'nt':
        _get_time = time.clock
    else:
        _get_time = time.time


class _ExtensionEntry(object):

    def __init__(self, extension, priority, trigger, call_before_training):
        self.extension = extension
        self.trigger = trigger
        self.priority = priority
        self.call_before_training = call_before_training


class Status(object):
    def __init__(self, epoch, iteration, epoch_size):
        self._epoch = epoch
        self._iteration = iteration
        self._epoch_size = epoch_size

    @property
    def epoch(self):
        return self._epoch

    @property
    def iteration(self):
        return self._iteration

    @property
    def epoch_detail(self):
        # epoch_size = len(self.engine.state.dataloader)
        return self._iteration/self._epoch_size

    @property
    def is_before_training(self):
        return self._iteration == 0


class ExtensionsManager(object):
    """
    Keeps track of the extensions and the current status
    """
    def __init__(self, max_epochs, extensions, out_dir='result'):
        self.stop_trigger = trigger_module.get_trigger((max_epochs, 'epoch'))
        self.observation = {}
        self.out = out_dir
        if not os.path.exists(self.out):
            os.makedirs(self.out)
        self.reporter = Reporter()

        self.max_epochs = max_epochs
        # Defer!
        self._start_time = None
        self._extensions = collections.OrderedDict()
        for ext in extensions:
            self.extend(ext)

    @property
    def elapsed_time(self):
        return _get_time()-self._start_time

    def start_extensions(self):
        exts = self._extensions
        extension_order = sorted(
            exts.keys(),
            key=lambda name: exts[name].priority, reverse=True)
        self.extensions = [(name, exts[name])
                           for name in extension_order]

        # invoke initializer of each extension
        for _, entry in self.extensions:
            initializer = getattr(entry.extension, 'initialize', None)
            finished = getattr(entry.trigger, 'finished', False)
            if initializer and not finished:
                initializer(self)

        # call extensions before training loop
        self.observation = {}
        with self.reporter.scope(self.observation):
            for name, entry in self.extensions:
                if entry.call_before_training:
                    entry.extension(self)

    def extend(self, extension, name=None, trigger=None, priority=None,
               *, call_before_training=False, **kwargs):
        """Registers an extension to the trainer.

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
        if name is None:
            name = getattr(extension, 'name', None)
            if name is None:
                name = getattr(extension, 'default_name', None)
                if name is None:
                    name = getattr(extension, '__name__', None)
                    if name is None:
                        raise TypeError('name is not given for the extension')
        if name == 'training':
            raise ValueError(
                'the name "training" is prohibited as an extension name')

        if trigger is None:
            trigger = getattr(extension, 'trigger', (1, 'iteration'))
        trigger = trigger_module.get_trigger(trigger)

        if priority is None:
            priority = getattr(
                extension, 'priority', extension_module.PRIORITY_READER)

        modified_name = name
        ordinal = 0
        while modified_name in self._extensions:
            ordinal += 1
            modified_name = '%s_%d' % (name, ordinal)

        extension.name = modified_name
        self._extensions[modified_name] = _ExtensionEntry(
            extension, priority, trigger, call_before_training)

    def get_extension(self, name):
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

    def run_extensions(self, epoch, iteration, epoch_size):
        self.status = Status(epoch, iteration, epoch_size)
        for name, entry in self.extensions:
            if entry.trigger(self):
                entry.extension(self)

    @contextlib.contextmanager
    def run_iteration(self, **kwargs):
        epoch = kwargs.pop('epoch')
        iteration = kwargs.pop('iteration')
        epoch_size = kwargs.pop('epoch_size')
        if self._start_time is None:
            self._start_time = _get_time()
        with self.reporter.scope(self.observation):
            try:
                yield
            finally:
                self.run_extensions(epoch, iteration, epoch_size)
