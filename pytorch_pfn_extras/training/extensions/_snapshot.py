import os

import torch
import torch.distributed

from pytorch_pfn_extras import logging
from pytorch_pfn_extras.training import extension


logger = logging._get_root_logger()


def _find_snapshot_files(fmt, path, fs):
    '''Only prefix and suffix match

    TODO(kuenishi): currently clean format string such as
    "snapshot{.iteration}.npz" can only be parsed, but tricky (or
    invalid) formats like "snapshot{{.iteration}}.npz" are hard to
    detect and to properly show errors, just ignored or fails so far.

    Args:
        fmt (str): format string to match with file names of
            existing snapshots, where prefix and suffix are
            only examined. Also, files' staleness is judged
            by timestamps. The default is metime.
        path (str): a directory path to search for snapshot files.

    Returns:
        A sorted list of pair of ``mtime, filename``, whose file
        name that matched the format ``fmt`` directly under ``path``.

    '''
    prefix = fmt.split('{')[0]
    suffix = fmt.split('}')[-1]

    matched_files = (file for file in fs.list(path)
                     if file.startswith(prefix) and file.endswith(suffix))

    def _prepend_mtime(f):
        t = fs.stat(os.path.join(path, f)).last_modified
        return (t, f)
    return sorted(_prepend_mtime(file) for file in matched_files)


def _find_latest_snapshot(fmt, path, fs):
    """Finds the latest snapshots in a directory

    Args:
        fmt (str): format string to match with file names of
            existing snapshots, where prefix and suffix are
            only examined. Also, files' staleness is judged
            by timestamps. The default is metime.
        path (str): a directory path to search for snapshot files.

    Returns:
        Latest snapshot file, in terms of a file that has newest
        ``mtime`` that matches format ``fmt`` directly under
        ``path``. If no such file found, it returns ``None``.

    """
    snapshot_files = _find_snapshot_files(fmt, path, fs)
    logger.debug('found snapshot files {}'.format(snapshot_files))
    if len(snapshot_files) > 0:
        _, filename = snapshot_files[-1]
        return filename
    return None


def _find_stale_snapshots(fmt, path, n_retains, fs):
    """Finds stale snapshots in a directory, retaining several files

    Args:
        fmt (str): format string to match with file names of
            existing snapshots, where prefix and suffix are
            only examined. Also, files' staleness is judged
            by timestamps. The default is metime.
        path (str): a directory path to search for snapshot files.
        n_retains (int): Number of snapshot files to retain
            through the cleanup. Must be a positive integer for any cleanup to
            take place.

    Returns:
        Generator that yields stale files that matches format
        ``fmt`` directly under ``path`` and with older ``mtime``,
        excluding newest ``n_retains`` files.

    """
    snapshot_files = _find_snapshot_files(fmt, path, fs)
    num_remove = len(snapshot_files) - n_retains
    if num_remove > 0:
        for _, filename in snapshot_files[:num_remove]:
            yield filename
    return


def snapshot_object(target, filename, savefun=None, **kwargs):
    """snapshot_object(target, filename, savefun=None, \
*, condition=None, writer=None, snapshot_on_error=False, \
n_retains=-1, autoload=False)

    Returns an extension to take snapshots of a given object.

    This extension serializes the given object and saves it to the output
    directory.

    This extension is called once per epoch by default. To take a
    snapshot at a different interval, a trigger object specifying the
    required interval can be passed along with this extension
    to the `extend()` method of the manager.

    The default priority is lower than that of most
    built-in extensions.

    Args:
        target: Object to serialize.
        filename (str): Name of the file into which the object is serialized.
            It can be a format string, where the manager object is passed to
            the :meth:`str.format` method. For example,
            ``'snapshot_{.iteration}'`` is converted to
            ``'snapshot_10000'`` at the 10,000th iteration.
        savefun: Function to save the object. It takes two arguments: the
            output file path and the object to serialize.
        condition: Condition object. It must be a callable object that returns
            boolean without any arguments. If it returns ``True``, the snapshot
            will be done.
            If not, it will be skipped. The default is a function that always
            returns ``True``.
        writer: Writer object.
            It must be a callable object.
            See below for the list of built-in writers.
            If ``savefun`` is other than ``None``, this argument must be
            ``None``. In that case, a
            :class:`~pytorch_pfn_extras.writing.SimpleWriter`
            object instantiated with specified ``savefun`` argument will be
            used.
        snapshot_on_error (bool): Whether to take a snapshot in case training
            loop has failed.
        n_retains (int): Number of snapshot files to retain
            through the cleanup. Must be a positive integer for any cleanup to
            take place. Automatic deletion of old snapshots only works when the
            filename is string.
        autoload (bool): With this enabled, the extension automatically
            finds the latest snapshot and loads the data to the target.
            Automatic loading only works when the filename is a string.
        saver_rank (int): If defined, the snapshot will be taken by only one
            rank when running in distributed mode and restored by all.
        transform_models (callable): If defined, function to apply to a model
            before obtaining its `state_dict`. Takes two parameters, the object
            name and the object itself.

    Returns:
        Snapshot extension object.

    .. seealso::

        - :meth:`pytorch_pfn_extras.training.extensions.snapshot`
    """

    return snapshot(target=target, filename=filename, savefun=savefun,
                    **kwargs)


def snapshot(savefun=None,
             filename='snapshot_iter_{.iteration}',
             *,
             target=None,
             condition=None,
             writer=None,
             snapshot_on_error=False,
             n_retains=-1,
             autoload=False,
             saver_rank=None,
             transform_models=None):
    """
    Returns a trainer extension to take snapshots of the trainer.

    This extension serializes the manager object and saves it to the output
    directory. It is used to support resuming the training loop from the saved
    state.

    This extension is called once per epoch by default. To take a
    snapshot at a different interval, a trigger object specifying the
    required interval can be passed along with this extension
    to the `extend()` method of the manager.

    The default priority is -100, which is lower than that of most
    built-in extensions.

    Args:
        savefun: Function to save the manager. It takes two arguments: the
            output file path and the manager object.
            It is :meth:`torch.save` by default.
            If ``writer`` is specified, this argument must be ``None``.
        filename (str): Name of the file into which the manager is serialized.
            It can be a format string, where the manager object is passed to
            the :meth:`str.format` method.
        target: Object to serialize. If it is not specified, it will
            be the manager object.
        condition: Condition object. It must be a callable object that returns
            boolean without any arguments. If it returns ``True``, the snapshot
            will be done.
            If not, it will be skipped. The default is a function that always
            returns ``True``.
        writer: Writer object.
            It must be a callable object.
            See below for the list of built-in writers.
            If ``savefun`` is other than ``None``, this argument must be
            ``None``. In that case, a
            :class:`~pytorch_pfn_extras.writing.SimpleWriter`
            object instantiated with specified ``savefun`` argument will be
            used.
        snapshot_on_error (bool): Whether to take a snapshot in case training
            loop has been failed.
        n_retains (int): Number of snapshot files to retain
            through the cleanup. Must be a positive integer for any cleanup to
            take place. Automatic deletion of old snapshots only works when the
            filename is string.
        autoload (bool): With this enabled, the extension
            automatically finds the latest snapshot and loads the data
            to the target.  Automatic loading only works when the
            filename is a string. It is assumed that snapshots are generated
            by :func:`torch.save` .
        saver_rank (int): If defined, the snapshot will be taken by only one
            rank when running in distributed mode and restored by all.
        transform_models (callable): If defined, function to apply to a model
            before obtaining its `state_dict`. Takes two parameters, the object
            name and the object itself.
    Returns:
        Snapshot extension object.

    .. testcode::
       :hide:

       from pytorch_pfn_extras import training
       class Model(torch.nn.Module):
           def __call__(self, x):
               return x

       model = Model()
       models = {'main': model}
       manager = training.ExtensionsManager(models, {}, 1, [])

    .. admonition:: Using asynchronous writers

        By specifying ``writer`` argument, writing operations can be made
        asynchronous, hiding I/O overhead of snapshots.

        >>> from pytorch_pfn_extras.training import extensions
        >>> from pytorch_pfn_extras import writing
        >>> writer = writing.ProcessWriter()
        >>> manager.extend(extensions.snapshot(writer=writer), \
trigger=(1, 'epoch'))

        To change the format, you can pass a saving
        function as ``savefun`` argument of the writer.

        >>> from pytorch_pfn_extras.training import extensions
        >>> from pytorch_pfn_extras import writing
        >>> writer = writing.ProcessWriter(
        ...     savefun=torch.save)
        >>> manager.extend(extensions.snapshot(writer=writer), \
trigger=(1, 'epoch'))

    This is the list of built-in snapshot writers.

        - :class:`pytorch_pfn_extras.writing.SimpleWriter`
        - :class:`pytorch_pfn_extras.writing.ThreadWriter`
        - :class:`pytorch_pfn_extras.writing.ProcessWriter`
        - :class:`pytorch_pfn_extras.writing.ThreadQueueWriter`
        - :class:`pytorch_pfn_extras.writing.ProcessQueueWriter`

    .. seealso::

        - :meth:`pytorch_pfn_extras.training.extensions.snapshot_object`
    """
    if savefun is not None and writer is not None:
        raise TypeError(
            'savefun and writer arguments cannot be specified together.')

    if saver_rank is None:
        return _Snapshot(
            target=target, condition=condition, writer=writer,
            filename=filename, snapshot_on_error=snapshot_on_error,
            n_retains=n_retains, autoload=autoload, savefun=savefun,
            transform_models=transform_models)
    return _DistributedSnapshot(
        target=target, condition=condition, writer=writer, filename=filename,
        snapshot_on_error=snapshot_on_error, n_retains=n_retains,
        autoload=autoload, saver_rank=saver_rank, savefun=savefun)


def _always_true():
    return True


class _Snapshot(extension.Extension):
    """An extension to take snapshots.

    This extension serializes the given object and saves it to the output
    directory.

    This extension is called once per epoch by default. To take a
    snapshot at a different interval, a trigger object specifying the
    required interval can be passed along with this extension
    to the `extend()` method of the manager.

    The default priority is -100, which is lower than that of most
    built-in extensions.
    """
    trigger = 1, 'epoch'
    priority = extension.PRIORITY_SNAPSHOT

    def __init__(
            self, target=None, condition=None, writer=None,
            filename='snapshot_iter_{.iteration}',
            snapshot_on_error=False, n_retains=-1, autoload=False,
            savefun=None,
            transform_models=None):
        if condition is None:
            condition = _always_true
        self._target = target
        self.filename = filename
        self.condition = condition
        self.writer = writer
        self._snapshot_on_error = snapshot_on_error
        self.n_retains = n_retains
        self.autoload = autoload
        self._savefun = savefun
        self._transform_models = transform_models

    def initialize(self, manager):
        target = manager if self._target is None else self._target
        outdir = manager.out
        writer = manager.writer if self.writer is None else self.writer
        self.writer = writer
        loaded_fn = None
        if self.autoload:
            # If ``autoload`` is on, this code scans the ``outdir``
            # for potential snapshot files by matching the file names
            # from ``filename`` format, picks up the latest one in
            # terms of mtime, and tries to load it it the target or
            # manager.
            loaded_fn = _find_latest_snapshot(self.filename, outdir, writer.fs)
            if loaded_fn:
                snapshot_file = writer.fs.open(os.path.join(outdir, loaded_fn))
                # As described above (at ``autoload`` option),
                # snapshot files to be autoloaded must be saved by
                # ``save_npz`` . In order to support general format,
                # we nned to first reconstruct the design of savefun
                # and loadfun.
                state = torch.load(snapshot_file,
                                   map_location=torch.device("cpu"))
                kwargs = {}
                if type(target) is dict:
                    for k in target:
                        target[k].load_state_dict(state[k], **kwargs)
                else:
                    target.load_state_dict(state, **kwargs)
                snapshot_file.close()

        if (hasattr(writer, '_add_cleanup_hook')
                and self.n_retains > 0
                and isinstance(self.filename, str)):
            # This block sets a method to automatic cleanup of stale
            # snapshots, when ``n_retains`` argument is positive
            # number. When the given snapshot writer is Chainer's
            # built-in writer, a cleanup method that is to be
            # triggered right after creation of new snapshot file, is
            # injected here.
            def _cleanup():
                files = _find_stale_snapshots(self.filename, outdir,
                                              self.n_retains, writer.fs)
                for file in files:
                    writer.fs.remove(os.path.join(outdir, file))

            writer._add_cleanup_hook(_cleanup)

        return loaded_fn

    def on_error(self, manager, exc, tb):
        super().on_error(manager, exc, tb)
        if self._snapshot_on_error:
            self._make_snapshot(manager)

    def __call__(self, manager):
        if self.condition():
            self._make_snapshot(manager)

    def _make_snapshot(self, manager):
        target = manager if self._target is None else self._target
        writer = manager.writer if self.writer is None else self.writer
        self.writer = writer
        # We need to get a dictionary with the state here
        kwargs = {}
        # If the user defines a transform_models function and a custom
        # target, he knows what he is doing so he should override state_dict
        # for his own target
        if self._transform_models is not None:
            kwargs['transform_models'] = self._transform_models

        if type(target) is dict:
            serialized_target = {
                k: v.state_dict(**kwargs) for k, v in target.items()}
        else:
            serialized_target = target.state_dict(**kwargs)
        filename = self.filename
        if callable(filename):
            filename = filename(manager)
        else:
            filename = filename.format(manager)
        outdir = manager.out
        writer(filename, outdir, serialized_target, savefun=self._savefun)

    def finalize(self):
        if hasattr(self.writer, 'finalize'):
            self.writer.finalize()


class _DistributedSnapshot(_Snapshot):
    """Trainer extension to take snapshots.

    This extension serializes the given object and saves it to the output
    directory.

    This extension is called once per epoch by default. To take a
    snapshot at a different interval, a trigger object specifying the
    required interval can be passed along with this extension
    to the `extend()` method of the trainer.

    The default priority is lower than that of most
    built-in extensions.
    """
    trigger = 1, 'epoch'
    priority = extension.PRIORITY_SNAPSHOT

    def __init__(
            self, target=None, condition=None, writer=None,
            filename='snapshot_iter_{.iteration}',
            snapshot_on_error=False, n_retains=-1, autoload=False,
            saver_rank=0, savefun=None, transform_models=None):
        super().__init__(target, condition, writer, filename,
                         snapshot_on_error, n_retains,
                         autoload, savefun, transform_models)
        # To support distributed snapshots
        if not torch.distributed.is_initialized():
            raise RuntimeError('The Distributed Snapshot extension',
                               ' requires torch.distributed to be initialized')
        self._saver_rank = saver_rank
        self._size = torch.distributed.get_world_size()
        self._rank = torch.distributed.get_rank()
        if not (0 <= saver_rank < self._size):
            raise ValueError('Distributed snapshot requires a saver rank'
                             ' in the range [0-{})'.format(self._size))

    def __call__(self, trainer):
        if self.condition():
            # on distributed environments only the designed rank
            # saves the snapshot
            if self._rank == self._saver_rank:
                self._make_snapshot(trainer)
            if self._size > 1:
                torch.distributed.barrier()
