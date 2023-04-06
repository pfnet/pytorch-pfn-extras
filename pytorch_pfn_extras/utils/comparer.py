import collections
import pathlib
import re
import threading
import weakref
import concurrent.futures
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence,
    Tuple, Type, Union,
)

import torch.nn
import torch.testing

import pytorch_pfn_extras
from pytorch_pfn_extras import handler as _handler_module
from pytorch_pfn_extras.handler import _logic
from pytorch_pfn_extras.training import _trainer
from pytorch_pfn_extras.training import manager as manager_module
from pytorch_pfn_extras.training import _evaluator
from pytorch_pfn_extras.training import trigger as trigger_module


_thread_local = threading.local()
_intermediate_prefix = "intermedaite:"


class _ComparableHandler(_handler_module.BaseHandler):
    def __init__(
            self,
            handler: _handler_module.BaseHandler,
            name: str,
            get_target_cb: Any,
            compare_cb: Any,
            trigger: Optional[trigger_module.Trigger] = None,
            *,
            dir: Optional[str] = None,
    ) -> None:
        self._handler = handler
        self._get_target_cb = get_target_cb
        self._compare_cb = compare_cb
        self.name = name
        self._trigger = trigger
        self._intermediate_values: Dict[int, Dict[str, Any]] = {}
        self._intermediate_counts: Dict[int, Dict[str, int]] = {}
        self._batch_idx: Optional[int] = None
        self._dir = dir
        self._epoch: Optional[int] = None
        self._entry_runtime = self._handler._entry_runtime  # type: ignore[attr-defined]

    def train_setup(self, trainer: _trainer._Trainer, loader: Any) -> None:
        self._handler.train_setup(trainer, loader)

    def train_epoch_begin(
            self, trainer: _trainer._Trainer, loader: Iterable[Any]) -> None:
        self._handler.train_epoch_begin(trainer, loader)
        _thread_local.handler = self
        self._epoch = trainer.epoch

    def train_epoch_end(self, trainer: _trainer._Trainer) -> None:
        self._handler.train_epoch_end(trainer)

    def train_validation_begin(
            self, trainer: _trainer._Trainer, evaluator: _evaluator.Evaluator) -> None:
        self._handler.train_validation_begin(trainer, evaluator)
        _thread_local.handler = self

    def train_validation_end(
            self, trainer: _trainer._Trainer, evaluator: _evaluator.Evaluator) -> None:
        self._handler.train_validation_end(trainer, evaluator)

    def train_step(
            self,
            trainer: _trainer._Trainer,
            batch_idx: int,
            batch: Any,
            complete_fn: Callable[[int, Any], None],
    ) -> None:
        self._batch_idx = batch_idx
        self._reset_intermediate_values()
        self._handler.train_step(trainer, batch_idx, batch, complete_fn)

    def train_post_step(
            self,
            trainer: _trainer.Trainer,
            batch_idx: int,
            batch: Any,
            outputs: Any
    ) -> None:
        class _ManagerProxy(manager_module._ManagerProxy):
            @property
            def iteration(self) -> int:
                # `Comparer._compare_targets` will be called
                # before `iteration` is incremented.
                return self._manager.iteration + 1

        manager = _ManagerProxy(trainer.manager)
        self._handler.train_post_step(trainer, batch_idx, batch, outputs)
        if self._trigger is None or self._trigger(manager):
            self._compare(trainer, batch_idx, outputs)

    def eval_setup(
            self, evaluator: _evaluator.Evaluator, loader: Iterable[Any]) -> None:
        self._handler.eval_setup(evaluator, loader)

    def eval_loop_begin(self, evaluator: _evaluator.Evaluator) -> None:
        self._handler.eval_loop_begin(evaluator)
        _thread_local.handler = self

    def eval_step(
            self,
            evaluator: _evaluator.Evaluator,
            batch_idx: int,
            batch: Any,
            complete_fn: Callable[[int, Any], None],
    ) -> None:
        self._batch_idx = batch_idx
        self._reset_intermediate_values()
        self._handler.eval_step(evaluator, batch_idx, batch, complete_fn)

    def eval_loop_end(self, evaluator: _evaluator.Evaluator) -> None:
        self._handler.eval_loop_end(evaluator)

    def eval_post_step(
            self,
            evaluator: _evaluator.Evaluator,
            batch_idx: int,
            batch: Any,
            outputs: Any,
    ) -> None:
        self._handler.eval_post_step(evaluator, batch_idx, batch, outputs)
        self._compare(evaluator, batch_idx, outputs)

    def _compare(self, engine: Any, batch_idx: int, outputs: Any) -> None:
        outputs = _logic._normalize_outputs(outputs)
        target = self._get_target_cb(self, engine, batch_idx, outputs)
        self._compare_cb(self.name, engine, batch_idx, target)

    def _reset_intermediate_values(self) -> None:
        assert self._batch_idx is not None
        self._intermediate_values[self._batch_idx] = {}
        self._intermediate_counts[self._batch_idx] = {}

    def _add_intermediate_value(self, name: str, value: torch.Tensor) -> None:
        assert self._batch_idx is not None
        values = self._intermediate_values[self._batch_idx]
        counts = self._intermediate_counts[self._batch_idx]
        value = value.detach()
        count = counts.get('name', 0)
        counts['name'] = count + 1
        name = _intermediate_prefix + name + f'_{count}'

        # Defer importing onnx for performance and to avoid linkage issues.
        import pytorch_pfn_extras.onnx
        if pytorch_pfn_extras.onnx.available:
            pytorch_pfn_extras.onnx.as_output(name, value)
        values[name] = value


def _overwrite_handler(engine: Any, *args: Any, **kwargs: Any) -> None:
    engine.handler = _ComparableHandler(engine.handler, *args, **kwargs)
    evaluator = getattr(engine, 'evaluator', None)
    if evaluator is not None:
        # For trainer with evaluator
        evaluator.handler = _ComparableHandler(evaluator.handler, *args, **kwargs)


_CompareFn = Callable[[str, str, str, Any, Any], None]
_Engine = Union[_trainer.Trainer, _evaluator.Evaluator]


def get_default_comparer(
        rtol: float = 1e-04,
        atol: float = 0,
        equal_nan: bool = True,
) -> _CompareFn:
    """Creates default comparer function.

    The created function will compare the outputs by using
    `torch.testing.assert_allclose` with specified options.

    Args:
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.
        equal_nan (bool): If ``True``, NaNs will be ignored.
    """
    def compare_fn(
            backend1: str, backend2: str, name: str,
            val1: Any, val2: Any) -> None:
        # TODO select the device where
        # the tensors will be compared?
        if isinstance(val1, torch.Tensor):
            val1 = val1.cpu().detach()
        if isinstance(val2, torch.Tensor):
            val2 = val2.cpu().detach()

        if pytorch_pfn_extras.requires("1.9.0"):
            assert_close = torch.testing.assert_close  # type: ignore[attr-defined]
        else:
            assert_close = torch.testing.assert_allclose  # type: ignore[assignment]

        assert_close(val1, val2, rtol=rtol, atol=atol, equal_nan=equal_nan)

    return compare_fn


_default_comparer = get_default_comparer()


def _compare_targets(
        compare_fn: _CompareFn,
        targets: Dict[str, Any],
        baseline: Optional[str],
        batch_idx: int,
) -> None:
    names = list(targets.keys())
    if baseline is None:
        baseline = names[0]
    keys = sorted(targets[baseline].keys())

    err_msg = ''
    for backend in set(names) - set([baseline]):
        for val_name in keys:
            out1 = targets[baseline][val_name]
            out2 = targets[backend][val_name]
            try:
                compare_fn(baseline, backend, val_name, out1, out2)
            except AssertionError as e:
                err_msg += (
                    f"Comparing '{baseline}' and '{backend}' in '{val_name}'\n"
                    f"{str(e)}\n")
    if err_msg:
        raise AssertionError(f'Batch: {batch_idx}\n' + str(err_msg))


class _ComparerBase:
    def __init__(
            self,
            engines: Mapping[str, _Engine],
            *,
            compare_fn: _CompareFn = _default_comparer,
            concurrency: Optional[int] = None,
    ) -> None:
        e_type = type(next(iter(engines.values())))
        if e_type not in (
            _trainer.Trainer,
            _evaluator.Evaluator,
        ):
            raise ValueError(f"Engine type {e_type} is not supported")
        if not all((isinstance(e, e_type) for e in engines.values())):
            raise ValueError("All the engines must be of the same type")

        self.engines = engines  # Need to wrap the handle with ours
        # If to_compare_key is None, then we compare all
        self.barrier = threading.Barrier(len(engines))
        self.report_lock = threading.Lock()
        self.compare_fn = compare_fn
        self._finalized = False
        self._semaphore = threading.Semaphore(
            len(engines) if concurrency is None else concurrency)
        self.targets: Dict[str, Dict[str, Any]] = {}
        self._iters: Dict[str, int] = {}
        # engines must be a dict
        for name, engine in engines.items():
            _overwrite_handler(engine, name, self._get_target, self.compare_targets)

    def _assert_incompatible_trigger(self, condition: bool) -> None:
        if not condition:
            raise ValueError('Engines have different triggers.')

    def run_engine(self, engine: _Engine, loaders: Any) -> None:
        try:
            self._semaphore.acquire()
            if isinstance(loaders, tuple):
                engine.run(*loaders)
            elif isinstance(loaders, dict):
                engine.run(**loaders)
            else:
                engine.run(loaders)
            with self.report_lock:
                self._finalized = True
                self._assert_incompatible_trigger(len(self.targets) == 0)
        except Exception:
            self.barrier.abort()
            raise
        finally:
            self._semaphore.release()

    def compare(self, loaders: Any, n_iters: Optional[int] = None) -> None:
        """Compares outputs.

        Args:
            loaders (dict of loaders):
                Data loaders used as input for each engine.
        """
        # n_iters is the number of iterations that we wait for
        # compare
        self.n_iters = n_iters
        self._iters = {k: 0 for k in self.engines.keys()}
        # We need to use a thread pool because is not easy at all to sync
        # the run method of different engines to compare every n iterations
        for name in self.engines.keys():
            if name not in loaders:
                raise KeyError(f"'{name}' is not in `loaders`")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self.engines)
        ) as executor:
            futures = []
            for name, engine in self.engines.items():
                futures.append(executor.submit(
                    self.run_engine, engine, loaders[name]))
            for future in concurrent.futures.as_completed(futures):
                future.result()

    def _get_target(
            self,
            handle: _handler_module.BaseHandler,
            engine: _Engine,
            batch_idx: int,
            outputs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError('Comparers must override _get_target')

    def compare_targets(
            self,
            name: str,
            engine: _Engine,
            batch_idx: int,
            target: Dict[str, Any],
    ) -> None:
        self._iters[name] += 1
        if (self.n_iters is None) or (self._iters[name] % self.n_iters == 0):
            # Save the outputs of this iteration
            with self.report_lock:
                self.targets[name] = target
                if len(self.targets.keys()) == len(self.engines.keys()):
                    # all outputs have been filled, lets compare and reset
                    _compare_targets(self.compare_fn, self.targets, None, batch_idx)
                    self.targets = {}
                self._assert_incompatible_trigger(not self._finalized)
            # Excplicitly synchronize
            self._semaphore.release()
            self.barrier.wait()
            self._semaphore.acquire()


class OutputsComparer(_ComparerBase):
    def __init__(
            self,
            engines: Mapping[str, _Engine],
            to_compare_keys: Optional[Sequence[str]] = None,
            *,
            compare_fn: _CompareFn = _default_comparer,
            concurrency: Optional[int] = None,
    ) -> None:
        """A class for comparison of iteration outputs.

        This class is mainly used to compare results between different devices.

        Args:
            engines (dict of Engines):
                Trainers or Evaluators to compare outputs.
            to_compare_keys (tuple of str, optional):
                A set of keys of output dict to compare.
            compare_fn (function):
                Comparison function. Default is ``get_default_comparer()``.
            concurrency (int, optional):
                The upper bound limit on the number of workers that run concurrently.
                If ``None``, inferred from the size of ``engines``.

        Examples:
            >>> trainer_cpu = ppe.engine.create_trainer(
                    model, optimizer, 1, device='cpu')
            >>> trainer_gpu = ppe.engine.create_trainer(
                    model, optimizer, 1, device='cuda:0')
            >>> comp = ppe.utils.comparer.OutputsComparer(
                    {"cpu": trainer_cpu, "gpu": trainer_gpu})
            >>> comp.compare({"cpu": loader, "gpu": loader}])
        """
        # If to_compare_key is None, then we compare all
        super().__init__(engines, compare_fn=compare_fn, concurrency=concurrency)
        self.to_compare_keys = to_compare_keys

    def _get_target(
            self,
            handle: _handler_module.BaseHandler,
            engine: _Engine,
            batch_idx: int,
            outputs: Any,
    ) -> Dict[str, Any]:
        keys = (
            self.to_compare_keys
            if self.to_compare_keys is not None
            else outputs.keys()
        )
        return {key: outputs[key] for key in keys}


class ModelComparer(_ComparerBase):
    def __init__(
            self,
            engines: Mapping[str, _Engine],
            to_compare_keys: Optional[Sequence[str]] = None,
            *,
            compare_fn: _CompareFn = _default_comparer,
            concurrency: Optional[int] = None,
    ):
        """A class for comparison of iteration model parameters.

        This class is mainly used to compare results between different devices.

        Args:
            engines (dict of Engines):
                Trainers or Evaluators to compare outputs.
            to_compare_keys (tuple of str, optional):
                A set of keys of model parameters to compare.
            compare_fn (function):
                Comparison function. Default is ``get_default_comparer()``.
            concurrency (int, optional):
                The upper bound limit on the number of workers that run concurrently.
                If ``None``, inferred from the size of ``engines``.

        Examples:
            >>> trainer_cpu = ppe.engine.create_trainer(
                    model, optimizer, 1, device='cpu')
            >>> trainer_gpu = ppe.engine.create_trainer(
                    model, optimizer, 1, device='cuda:0')
            >>> comp = ppe.utils.comparer.ModelComparer(
                    {"cpu": trainer_cpu, "gpu": trainer_gpu})
            >>> comp.compare({"cpu": loader, "gpu": loader}])
        """
        # If to_compare_key is None, then we compare all
        super().__init__(engines, compare_fn=compare_fn, concurrency=concurrency)
        self.to_compare_keys = to_compare_keys
        self._preprocessed_keys: Optional[List[str]] = None

    def _preprocess_keys(self, sdict: Dict[str, Any]) -> None:
        if self.to_compare_keys is None:
            self._preprocessed_keys = list(sdict.keys())
        else:
            self._preprocessed_keys = []
            for tc_k in self.to_compare_keys:
                matched = False
                for sd_k in sdict.keys():
                    if re.match(tc_k, sd_k) is not None:
                        self._preprocessed_keys.append(sd_k)
                        matched = True
                if not matched:
                    raise ValueError(
                        f'didnt find a match for {tc_k} in the model')

    def _get_target(
            self,
            handle: _handler_module.BaseHandler,
            engine: _Engine,
            batch_idx: int,
            outputs: Any,
    ) -> Dict[str, Any]:
        sdict = engine.models['main'].state_dict()
        if self._preprocessed_keys is None:
            self._preprocess_keys(sdict)
        assert self._preprocessed_keys is not None
        return {key: sdict[key] for key in self._preprocessed_keys}


# New comparer interface

def _filter(
        keys: Union[bool, str, Sequence[str]],
        get_dict: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    if keys is False:
        return {}
    if keys is True:
        return get_dict()

    if isinstance(keys, str):
        keys = (keys,)
    if isinstance(keys, (tuple, list)):
        if len(keys) == 0:
            return {}
        sdict = get_dict()
        ret = {}
        for tc_k in keys:
            for sd_k in sdict.keys():
                if re.match(tc_k, sd_k) is not None:
                    ret[sd_k] = sdict[sd_k]
                    break
            else:
                raise ValueError(f'didnt find a match for {tc_k} in the model')
        return ret

    raise ValueError(f'Unsupported type: {type(keys)}')


class Comparer:

    def __init__(
            self,
            *,
            trigger: Optional[trigger_module.TriggerLike] = None,
            compare_fn: _CompareFn = _default_comparer,
            concurrency: Optional[int] = None,
            outputs: Union[bool, str, Sequence[str]] = True,
            params: Union[bool, str, Sequence[str]] = False,
            baseline: Optional[str] = None,
    ) -> None:
        """A class for comparison of iteration outputs and model parameters.

        This class is mainly used to compare results between different devices.

        Args:
            trigger (Trigger):
                Trigger object that determines when to compare values.
            compare_fn (function):
                Comparison function. Default is ``get_default_comparer()``.
            concurrency (int, optional):
                The upper bound limit on the number of workers that run concurrently.
                If ``None``, inferred from the size of ``engines``.
            outputs (tuple of str or bool):
                A set of keys of output dict to compare.
            params (tuple of str or bool):
                A set of keys of model parameters to compare.
            baseline (str, optional):
                The baseline engine that is assumed to be correct.

        Examples:
            >>> trainer_cpu = ppe.engine.create_trainer(
                    model, optimizer, 1, device='cpu')
            >>> trainer_gpu = ppe.engine.create_trainer(
                    model, optimizer, 1, device='cuda:0')
            >>> comp = ppe.utils.comparer.Comparer()
            >>> comp.add_engine("cpu", engine_cpu, train_1, eval_1)
            >>> comp.add_engine("gpu", engine_gpu, train_2, eval_2)
            >>> comp.compare()
        """
        self._engine_type: Optional[Type[_Engine]] = None
        self._engines: Dict[
            str, Tuple[Union[_Engine, _LoadDumpsEngine], Any, Any]
        ] = collections.OrderedDict()
        self._compare_fn = compare_fn
        self._targets: Dict[str, Dict[str, Any]] = {}
        self._output_keys = outputs
        self._param_keys = params
        self._baseline = baseline
        self._finalized = False
        self._concurrency = concurrency  # Upper limit of semaphore size
        # Sempaphore for training step execution
        self._semaphore: Optional[threading.Semaphore] = None
        # Synchronizes iteration timing
        self._barrier: Optional[threading.Barrier] = None
        self._report_lock = threading.Lock()  # Locks `Comparer._get_target`
        self._count = 0

        if trigger is None:
            self._trigger = trigger_module.get_trigger((1, "epoch"))
        else:
            self._engine_type = _trainer.Trainer
            self._trigger = trigger_module.get_trigger(trigger)

    def _get_target(
            self,
            handler: _ComparableHandler,
            engine: _Engine,
            batch_idx: int,
            outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        targets = {}
        outputs = _filter(self._output_keys, lambda: outputs)
        targets.update({
            k if k.startswith(_intermediate_prefix) else 'output:' + k: v
            for k, v in outputs.items()})
        targets.update(handler._intermediate_values.pop(batch_idx))

        params = _filter(self._param_keys, engine.models['main'].state_dict)
        targets.update({'param:' + k: v for k, v in params.items()})
        return targets

    def _assert_incompatible_trigger(self, condition: bool) -> None:
        if not condition:
            raise ValueError("Engines have different triggers.")

    def _get_filename(
            self, engine: _Engine, handler_name: str, batch_idx: int) -> str:
        name = f'dump_{self._count:08}'
        name += '_' + type(engine).__name__
        orig_engine, _, _ = self._engines[handler_name]
        epoch = orig_engine.handler._epoch  # type: ignore
        if epoch is not None:
            name += f'_epoch_{epoch}'
        name += f'_iter_{batch_idx}'
        return name

    def _compare_targets(
            self,
            name: str,
            engine: _Engine,
            batch_idx: int,
            target: Dict[str, Any],
    ) -> None:
        # Save the outputs of this iteration
        with self._report_lock:
            self._targets[name] = target
            if len(self._targets.keys()) == len(self._engines.keys()):
                # all outputs have been filled, lets compare and reset
                _compare_targets(
                    self._compare_fn, self._targets, self._baseline, batch_idx)
                self._targets = {}
                self._count += 1
            self._assert_incompatible_trigger(not self._finalized)

        # Excplicitly synchronize
        assert self._semaphore is not None
        assert self._barrier is not None
        self._semaphore.release()
        try:
            self._barrier.wait()
        finally:
            self._semaphore.acquire()

    def add_engine(
            self,
            name: str,
            engine: _Engine,
            *args: Any,
            **kwargs: Any,
    ) -> None:
        """Add an engine to compare variables.

        Args:
            name (str):
                Engine name.
            engine (Trainer or Evaluator):
                An engine to compare variables.
            *args and **kwargs:
                Arguments passed to ``engine.run``.
        """
        type_engine = type(engine)

        if type_engine not in (_trainer.Trainer, _evaluator.Evaluator):
            raise ValueError(f"Engine type {type_engine} is not supported")

        if self._engine_type is None:
            self._engine_type = type_engine
        elif type_engine != self._engine_type:
            raise ValueError("All the engines must be of the same type")

        if name in self._engines.keys():
            raise ValueError(f"Engine named {name} already registered")

        _overwrite_handler(
            engine, name, self._get_target, self._compare_targets, self._trigger)

        self._engines[name] = engine, args, kwargs

    def add_dump(self, name: str, dir: str) -> None:
        """Add an engine to compare variables.

        Args:
            name (str):
                The name of dump.
            dir (str):
                The directory that the results are saved to.
        """
        engine = _LoadDumpsEngine(self, name, dir)
        self._engines[name] = engine, (), {}

    def _dump_targets(
            self,
            name: str,
            engine: _Engine,
            batch_idx: int,
            target: Dict[str, Any],
    ) -> None:
        name = self._get_filename(engine, name, batch_idx)
        assert isinstance(engine.handler, _ComparableHandler)
        torch.save(target, f'{engine.handler._dir}/{name}')
        self._count += 1

    def dump(self, engine: _Engine, dir: str, *args: Any, **kwargs: Any) -> None:
        """Add an engine to compare variables.

        Args:
            engine (Trainer or Evaluator):
                An engine to compare variables.
            dir (str):
                Name of the directory that the results are saved to.
            *args and **kwargs:
                Arguments passed to ``engine.run``.
        """
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

        self._count = 0
        name = '__dump'
        _overwrite_handler(
            engine, name, self._get_target, self._dump_targets,
            self._trigger, dir=dir)

        self._engines[name] = engine, args, {}
        engine.run(*args, **kwargs)
        self._engines.pop(name)

        assert isinstance(engine.handler, _ComparableHandler)
        engine.handler = engine.handler._handler

    def _run_engine(self, engine: _Engine, args: Any, kwargs: Any) -> None:
        assert self._semaphore is not None
        assert self._barrier is not None
        self._semaphore.acquire()
        try:
            engine.run(*args, **kwargs)
            with self._report_lock:
                self._finalized = True
                self._assert_incompatible_trigger(len(self._targets) == 0)
        except Exception:
            self._barrier.abort()
            raise
        finally:
            self._semaphore.release()

    def compare(self) -> None:
        """Compares outputs.
        """
        self._count = 0
        n_workers = len(self._engines)
        self._barrier = threading.Barrier(n_workers)
        self._semaphore = threading.Semaphore(
            n_workers if self._concurrency is None else self._concurrency)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for _, (engine, args, kwargs) in self._engines.items():
                futures.append(executor.submit(self._run_engine, engine, args, kwargs))  # type: ignore[arg-type]
            for future in concurrent.futures.as_completed(futures):
                future.result()


def intermediate_value(name: str, value: torch.Tensor) -> None:
    if not hasattr(_thread_local, 'handler'):
        return
    _thread_local.handler._add_intermediate_value(name, value)


class _LoadDumpsEngine:
    def __init__(self, comparer: Comparer, name: str, dir: str) -> None:
        self.name = name
        self._comparer_ref = weakref.ref(comparer)
        self._dir = dir

    def run(self) -> None:
        comparer = self._comparer_ref()
        assert comparer is not None
        for path in sorted(pathlib.Path(self._dir).iterdir()):
            filename = path.name
            if filename.startswith('dump_'):
                target = torch.load(path)  # type: ignore[no-untyped-call]
                iter_str = '_iter_'
                pos = filename.find(iter_str)
                batch_idx = int(filename[pos + len(iter_str):])
                comparer._compare_targets(
                    self.name, None, batch_idx, target)  # type: ignore
