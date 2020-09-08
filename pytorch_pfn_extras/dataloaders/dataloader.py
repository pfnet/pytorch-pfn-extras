r"""Definition of the DataLoader and associated iterators
that subclass _BaseDataLoaderIter

To support these two classes, in `./_utils` we define many utility methods and
functions to be run in multiprocessing. E.g., the data loading worker loop is
in `./utils/worker.py`.
"""

import threading
import itertools
import warnings

import multiprocessing as python_multiprocessing
import torch
import torch.multiprocessing as multiprocessing
from torch._utils import ExceptionWrapper
from torch._six import queue, string_classes

from torch.utils.data import (
    IterableDataset,
    Sampler,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
)
from torch.utils.data import _utils
from pytorch_pfn_extras.dataloaders._utils import worker


get_worker_info = worker.get_worker_info

# This function used to be defined in this file. However, it was moved to
# _utils/collate.py. Although it is rather hard to access this from user land
# (one has to explicitly directly `import torch.utils.data.dataloader`), there
# probably is user code out there using it. This aliasing maintains BC in this
# aspect.
default_collate = _utils.collate.default_collate


class _DatasetKind(object):
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(
                dataset, auto_collation, collate_fn, drop_last
            )
        else:
            return _utils.fetch._IterableDatasetFetcher(
                dataset, auto_collation, collate_fn, drop_last
            )


class _InfiniteConstantSampler(Sampler):
    r"""Analogous to ``itertools.repeat(None, None)``.
    Used as sampler for :class:`~torch.utils.data.IterableDataset`.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self):
        super(_InfiniteConstantSampler, self).__init__(None)

    def __iter__(self):
        while True:
            yield None


class DataLoader(object):
    r"""
    Data loader. Combines a dataset and a sampler, and provides an
    iterable over the given dataset.

    The :class:`~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation)
    and memory pinning.

    See :py:mod:`torch.utils.data` documentation page for more details.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`,
            but returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
            and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main
            process. (default: ``0``)
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory (bool, optional): If ``True``, the data loader will copy
            Tensors into CUDA pinned memory before returning them.
            If your data elements are a custom type, or your
            :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        drop_last (bool, optional): set to ``True`` to drop the last
            incomplete batch, if the dataset size is not divisible by
            the batch size. If ``False`` and the size of dataset is
            not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value
            for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (callable, optional): If not ``None``, this will
            be called on each worker subprocess with the worker id
            (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
        persistent_workers (bool, optional): If ``True``, the data loader
            will not shutdown the worker processes after a dataset has
            been consumed once. This allows to maintain the workers
            `Dataset` instances alive. (default: ``False``)


    .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                 cannot be an unpicklable object, e.g., a lambda function. See
                 :ref:`multiprocessing-best-practices` on more details related
                 to multiprocessing in PyTorch.

    .. note:: ``len(dataloader)`` heuristic is based on the length of the
                sampler used.
              When :attr:`dataset` is an
              :class:`~torch.utils.data.IterableDataset`, ``len(dataset)``
              (if implemented) is returned instead, regardless
              of multi-process loading configurations, because PyTorch trust
              user :attr:`dataset` code in correctly handling multi-process
              loading to avoid duplicate data. See `Dataset Types`_ for more
              details on these two types of datasets and how
              :class:`~torch.utils.data.IterableDataset` interacts
              with `Multi-process data loading`_.
    """

    __initialized = False

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        persistent_workers=False,
    ):
        torch._C._log_api_usage_once("python.data_loader")

        if num_workers < 0:
            raise ValueError(
                "num_workers option should be non-negative; "
                "use num_workers=0 to disable multiprocessing."
            )

        if timeout < 0:
            raise ValueError("timeout option should be non-negative")

        self.dataset = dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context

        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            if shuffle is not False:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "shuffle option, but got shuffle={}".format(shuffle)
                )
            elif sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "sampler option, but got sampler={}".format(sampler)
                )
            elif batch_sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "batch_sampler option, but got batch_sampler={}".format(
                        batch_sampler
                    )
                )
        else:
            self._dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError(
                "sampler option is mutually exclusive with " "shuffle"
            )

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler option is mutually exclusive "
                    "with batch_size, shuffle, sampler, and "
                    "drop_last"
                )
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError(
                    "batch_size=None option disables auto-batching "
                    "and is mutually exclusive with drop_last"
                )

        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                # See NOTE [ Custom Samplers and IterableDataset ]
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    sampler = RandomSampler(dataset, generator=generator)
                else:
                    sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.__initialized = True
        self._IterableDataset_len_called = (
            None  # See NOTE [ IterableDataset and __len__ ]
        )

        self._persistent_workers = persistent_workers
        self._iterator = None

    def _get_iterator(self):
        if self.num_workers == 0:
            iterator = _SingleProcessDataLoaderIter(self)
        else:
            iterator = _MultiProcessingDataLoaderIter(self)
        return iterator

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if not multiprocessing._supports_context:
                    raise ValueError(
                        "multiprocessing_context relies on Python >= 3.4, with"
                        " support for different start methods"
                    )

                if isinstance(multiprocessing_context, string_classes):
                    valid_start_methods = (
                        multiprocessing.get_all_start_methods()
                    )
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            (
                                "multiprocessing_context option "
                                "should specify a valid start method in {} "
                                " but got multiprocessing_context={}"
                            ).format(
                                valid_start_methods, multiprocessing_context
                            )
                        )
                    multiprocessing_context = multiprocessing.get_context(
                        multiprocessing_context
                    )

                if not isinstance(
                    multiprocessing_context,
                    python_multiprocessing.context.BaseContext,
                ):
                    raise TypeError(
                        (
                            "multiprocessing_context should be a valid context"
                            " object or a string specifying the start method"
                            " but got multiprocessing_context={}"
                        ).format(multiprocessing_context)
                    )
            else:
                raise ValueError(
                    (
                        "multiprocessing_context can only be used with "
                        "multi-process loading (num_workers > 0), but got "
                        "num_workers={}"
                    ).format(self.num_workers)
                )

        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val):
        if self.__initialized and attr in (
            "batch_size",
            "batch_sampler",
            "sampler",
            "drop_last",
            "dataset",
        ):
            raise ValueError(
                "{} attribute should not be set after {} is "
                "initialized".format(attr, self.__class__.__name__)
            )

        super(DataLoader, self).__setattr__(attr, val)

    def __iter__(self):
        # When using a single worker the returned iterator should be
        # created everytime to avoid reseting its state
        # However, in the case of a multiple workers iterator
        # the iterator is only created once in the lifetime of the
        # DataLoader object so that workers can be reused
        if self._persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self):
        if self._dataset_kind == _DatasetKind.Iterable:
            length = self._IterableDataset_len_called = len(self.dataset)
            if self.batch_size is not None:
                from math import ceil

                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = ceil(length / self.batch_size)
            return length
        else:
            return len(self._index_sampler)


class _BaseDataLoaderIter(object):
    def __init__(self, loader):
        self._dataset = loader.dataset
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        self._pin_memory = loader.pin_memory and torch.cuda.is_available()
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = (
            torch.empty((), dtype=torch.int64)
            .random_(generator=loader.generator)
            .item()
        )
        self._persistent_workers = loader._persistent_workers
        self._num_yielded = 0

    def _reset(self, loader, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called

    def __iter__(self):
        return self

    def _next_index(self):
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        raise NotImplementedError

    def __next__(self):
        data = self._next_data()
        self._num_yielded += 1
        if (
            self._dataset_kind == _DatasetKind.Iterable
            and self._IterableDataset_len_called is not None
            and self._num_yielded > self._IterableDataset_len_called
        ):
            warn_msg = (
                "Length of IterableDataset {} was reported to be {} "
                "(when accessing len(dataloader)), but {} "
                "samples have been fetched. "
            ).format(
                self._dataset,
                self._IterableDataset_len_called,
                self._num_yielded,
            )
            if self._num_workers > 0:
                warn_msg += (
                    "For multiprocessing data-loading, this could be caused "
                    "by not properly configuring the "
                    "IterableDataset replica at each worker. Please see "
                    "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples."  # NOQA
                )
            warnings.warn(warn_msg)
        return data

    next = __next__  # Python 2 compatibility

    def __len__(self):
        return len(self._index_sampler)

    def __getstate__(self):
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError(
            "{} cannot be pickled", self.__class__.__name__
        )


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self._dataset,
            self._auto_collation,
            self._collate_fn,
            self._drop_last,
        )

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super(_MultiProcessingDataLoaderIter, self).__init__(loader)

        assert self._num_workers > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn
        self._worker_queue_idx_cycle = itertools.cycle(
            range(self._num_workers)
        )
        self._worker_result_queue = multiprocessing_context.Queue()
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []
        self._workers = []
        # A list of booleans representing whether each worker still has work to
        # do, i.e., not having exhausted its iterable dataset object. It always
        # contains all `True`s if not using an iterable-style dataset
        # (i.e., if kind != Iterable).
        self._workers_status = []
        for i in range(self._num_workers):
            index_queue = multiprocessing_context.Queue()
            # index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=worker._worker_loop,
                args=(
                    self._dataset_kind,
                    self._dataset,
                    index_queue,
                    self._worker_result_queue,
                    self._workers_done_event,
                    self._auto_collation,
                    self._collate_fn,
                    self._drop_last,
                    self._base_seed + i,
                    self._worker_init_fn,
                    i,
                    self._num_workers,
                    self._persistent_workers,
                ),
            )
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)
            self._workers_status.append(True)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()
            self._data_queue = queue.Queue()
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(
                    self._worker_result_queue,
                    self._data_queue,
                    torch.cuda.current_device(),
                    self._pin_memory_thread_done_event,
                ),
            )
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        _utils.signal_handling._set_worker_pids(
            id(self), tuple(w.pid for w in self._workers)
        )
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)

    def _reset(self, loader, first_iter=False):
        super()._reset(loader, first_iter)
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        self._task_info = {}
        self._tasks_outstanding = 0
        # always equal to count(v for v in task_info.values() if len(v) == 1)
        self._workers_status = [True for i in range(self._num_workers)]
        # We resume the prefetching in case it was enabled
        if not first_iter:
            # The data queue might have older requests pending from a previous
            # iteration, lets delete them
            while not self._data_queue.empty():
                self._data_queue.get()
            for idx in range(self._num_workers):
                self._index_queues[idx].put(worker._ResumeIteration())
        # prime the prefetch loop
        for _ in range(2 * self._num_workers):
            self._try_put_index()

    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed. Note that this is the only mechanism for Windows
            # to detect worker failures.
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._shutdown_worker(worker_id)
            if len(failed_workers) > 0:
                pids_str = ", ".join(str(w.pid) for w in failed_workers)
                raise RuntimeError(
                    "DataLoader worker (pid(s) {}) exited unexpectedly".format(
                        pids_str
                    )
                )
            if isinstance(e, queue.Empty):
                return (False, None)
            import tempfile
            import errno

            try:
                # Raise an exception if we are this close to the FDs limit.
                # Apparently, trying to open only one file is not a sufficient
                # test.
                # See NOTE [ DataLoader on Linux and open files limit ]
                fds_limit_margin = 10
                fs = [  # NOQA
                    tempfile.NamedTemporaryFile()
                    for i in range(fds_limit_margin)
                ]
            except OSError as e:
                if e.errno == errno.EMFILE:
                    raise RuntimeError(
                        "Too many open files. Communication with the"
                        " workers is no longer possible. Please increase the"
                        " limit using `ulimit -n` in the shell or change the"
                        " sharing strategy by calling"
                        " `torch.multiprocessing.set_sharing_strategy('file_system')`"  # NOQA
                        " at the beginning of your code"
                    )
            raise

    def _get_data(self):
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError(
                    "DataLoader timed out after {} seconds".format(
                        self._timeout
                    )
                )
        elif self._pin_memory:
            while self._pin_memory_thread.is_alive():
                success, data = self._try_get_data()
                if success:
                    return data
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError("Pin memory thread exited unexpectedly")
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _next_data(self):
        while True:
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if (
                    len(info) == 2 or self._workers_status[worker_id]
                ):  # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(
                    data, worker._IterableDatasetStopIteration
                ):
                    self._shutdown_worker(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._process_data(data)

    def _try_put_index(self):
        assert self._tasks_outstanding < 2 * self._num_workers
        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(
            self._num_workers
        ):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index()
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data

    def _shutdown_worker(self, worker_id):
        # Mark a worker as having finished its work and dead, e.g., due to
        # exhausting an `IterableDataset`. This should be used only when this
        # `_MultiProcessingDataLoaderIter` is going to continue running.

        assert self._workers_status[worker_id]

        # Signal termination to that specific worker.
        q = self._index_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current
        # process.
        q.put(None)

        self._workers_status[worker_id] = False

    def _shutdown_workers(self):
        python_exit_status = _utils.python_exit_status
        if python_exit_status is True or python_exit_status is None:
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self._shutdown:
            self._shutdown = True
            try:
                if hasattr(self, "_pin_memory_thread"):
                    self._pin_memory_thread_done_event.set()
                    self._worker_result_queue.put((None, None))
                    self._pin_memory_thread.join()
                    self._worker_result_queue.cancel_join_thread()
                    self._worker_result_queue.close()

                # Exit workers now.
                self._workers_done_event.set()
                for worker_id in range(len(self._workers)):
                    if (
                        self._persistent_workers
                        or self._workers_status[worker_id]
                    ):
                        self._shutdown_worker(worker_id)
                for w in self._workers:
                    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                    if w.is_alive():
                        w.terminate()
                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                if self._worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False

    def __del__(self):
        self._shutdown_workers()
