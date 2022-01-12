# mypy: ignore-errors

import collections
import json
import re
import threading
import concurrent.futures
from typing import Any, Callable, Dict, Sequence, Union

import torch.testing

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras import engine as _engine_module
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
            self, handler, name, get_target_cb, compare_cb, trigger=None, *, dir=None):
        self._handler = handler
        self._get_target_cb = get_target_cb
        self._compare_cb = compare_cb
        self.name = name
        self._trigger = trigger
        self._intermediate_values = {}
        self._intermediate_counts = {}
        self._batch_idx = None
        self._dir = dir
        self._epoch = None

    def convert_batch(self, args):
        return self._handler.convert_batch(args)

    def train_setup(self, trainer, loader):
        self._handler.train_setup(trainer, loader)

    def train_epoch_begin(self, trainer, loader):
        self._handler.train_epoch_begin(trainer, loader)
        _thread_local.handler = self
        self._epoch = trainer.epoch

    def train_epoch_end(self, trainer):
        self._handler.train_epoch_end(trainer)
        self._epoch = None

    def train_validation_begin(self, trainer, evaluator):
        self._handler.train_validation_begin(trainer, evaluator)
        _thread_local.handler = self

    def train_validation_end(self, trainer, evaluator):
        self._handler.train_validation_end(trainer, evaluator)

    def train_step(self, trainer, batch_idx, batch, complete_fn):
        self._batch_idx = batch_idx
        self._reset_intermediate_values()
        self._handler.train_step(trainer, batch_idx, batch, complete_fn)

    def train_post_step(self, trainer, batch_idx, batch, outputs):
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

    def eval_setup(self, evaluator, loader):
        return self._handler.eval_setup(evaluator, loader)

    def eval_loop_begin(self, evaluator):
        self._handler.eval_loop_begin(evaluator)
        _thread_local.handler = self

    def eval_step(self, evaluator, batch_idx, batch, complete_fn):
        self._batch_idx = batch_idx
        self._reset_intermediate_values()
        self._handler.eval_step(evaluator, batch_idx, batch, complete_fn)

    def eval_loop_end(self, evaluator):
        self._handler.eval_loop_end(evaluator)

    def eval_post_step(self, evaluator, batch_idx, batch, outputs):
        self._handler.eval_post_step(evaluator, batch_idx, batch, outputs)
        self._compare(evaluator, batch_idx, outputs)

    def _compare(self, engine, batch_idx, outputs):
        outputs = _logic._normalize_outputs(outputs)
        target = self._get_target_cb(self, engine, batch_idx, outputs)
        self._compare_cb(self, engine, batch_idx, target)

    def _reset_intermediate_values(self) -> None:
        self._intermediate_values[self._batch_idx] = {}
        self._intermediate_counts[self._batch_idx] = {}

    def _add_intermediate_value(self, name: str, value: torch.Tensor) -> None:
        if self._intermediate_values is None:
            return
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


def _overwrite_handler(engine, *args, **kwargs) -> None:
    engine.handler = _ComparableHandler(engine.handler, *args, **kwargs)
    evaluator = getattr(engine, 'evaluator', None)
    if evaluator is not None:
        # For trainer with evaluator
        evaluator.handler = _ComparableHandler(evaluator.handler, *args, **kwargs)


def get_default_comparer(rtol=1e-04, atol=0, equal_nan=True):
    """Creates default comparer function.

    The created function will compare the outputs by using
    `torch.testing.assert_allclose` with specified options.

    Args:
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.
        equal_nan (bool): If ``True``, NaNs will be ignored.
    """
    def compare_fn(backend1: str, backend2: str, name: str, val1: Any, val2: Any):
        # TODO select the device where
        # the tensors will be compared?
        if isinstance(val1, torch.Tensor):
            val1 = val1.cpu().detach()
        if isinstance(val2, torch.Tensor):
            val2 = val2.cpu().detach()
        torch.testing.assert_allclose(
            val1, val2, rtol=rtol, atol=atol, equal_nan=equal_nan)

    return compare_fn


_default_comparer = get_default_comparer()


def _compare_targets(compare_fn, targets, baseline, batch_idx):
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
            self, engines, *,
            compare_fn=_default_comparer,
            concurrency=None,
    ):
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
        self.targets = {}
        self._iters = {}
        # engines must be a dict
        for name, engine in engines.items():
            _overwrite_handler(engine, name, self._get_target, self.compare_targets)

    def _assert_incompatible_trigger(self, condition):
        if not condition:
            raise ValueError('Engines have different triggers.')

    def run_engine(self, engine, loaders):
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

    def compare(self, loaders, n_iters=None):
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

    def _get_target(self, handle, models, outputs):
        raise NotImplementedError('Comparers must override _get_target')

    def compare_targets(self, handler, engine, batch_idx, target):
        self._iters[handler.name] += 1
        if (self.n_iters is None) or (self._iters[handler.name] % self.n_iters == 0):
            # Save the outputs of this iteration
            with self.report_lock:
                self.targets[handler.name] = target
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
            self, engines, to_compare_keys=None, *,
            compare_fn=_default_comparer,
            concurrency=None,
    ):
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

    def _get_target(self, handle, engine, batch_idx, outputs):
        keys = (
            self.to_compare_keys
            if self.to_compare_keys is not None
            else outputs.keys()
        )
        return {key: outputs[key] for key in keys}


class ModelComparer(_ComparerBase):
    def __init__(
            self, engines, to_compare_keys=None, *,
            compare_fn=_default_comparer,
            concurrency=None,
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
        self._preprocessed_keys = None

    def _preprocess_keys(self, sdict):
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

    def _get_target(self, handle, engine, batch_idx, outputs):
        sdict = engine.models['main'].state_dict()
        if self._preprocessed_keys is None:
            self._preprocess_keys(sdict)
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
            trigger=None,
            compare_fn=_default_comparer,
            concurrency=None,
            outputs=True,
            params=False,
            baseline=None,
    ):
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
        self._engine_type = None
        self._engines = collections.OrderedDict()
        self._compare_fn = compare_fn
        self._targets = {}
        self._output_keys = outputs
        self._param_keys = params
        self._baseline = baseline
        self._finalized = False
        self._concurrency = concurrency  # Upper limit of semaphore size
        self._semaphore = None  # Sempaphore for training step execution
        self._barrier = None  # Synchronizes iteration timing
        self._report_lock = threading.Lock()  # Locks `Comparer._get_target`

        if trigger is None:
            self._trigger = trigger_module.get_trigger((1, "epoch"))
        else:
            self._engine_type = _trainer.Trainer
            self._trigger = trigger_module.get_trigger(trigger)

    def _get_target(self, handler, engine, batch_idx, outputs):
        targets = {}
        outputs = _filter(self._output_keys, lambda: outputs)
        targets.update({
            k if k.startswith(_intermediate_prefix) else 'output:' + k: v
            for k, v in outputs.items()})
        targets.update(handler._intermediate_values.pop(batch_idx))

        params = _filter(self._param_keys, engine.models['main'].state_dict)
        targets.update({'param:' + k: v for k, v in params.items()})
        return targets

    def _assert_incompatible_trigger(self, condition):
        if not condition:
            raise ValueError("Engines have different triggers.")

    @staticmethod
    def _get_filename(engine, batch_idx):
        name = type(engine).__name__
        epoch = engine.handler._epoch
        if epoch is not None:
            name += f'_epoch_{epoch}'
        name += f'_iter_{batch_idx}'
        return name

    def _compare_targets(self, handler, engine, batch_idx, target):
        # Save the outputs of this iteration
        with self._report_lock:
            self._targets[handler.name] = target
            if len(self._targets.keys()) == len(self._engines.keys()):
                # all outputs have been filled, lets compare and reset
                _compare_targets(
                    self._compare_fn, self._targets, self._baseline, batch_idx)
                self._targets = {}
            self._assert_incompatible_trigger(not self._finalized)

        # Excplicitly synchronize
        self._semaphore.release()
        try:
            self._barrier.wait()
        finally:
            self._semaphore.acquire()

    def add_engine(self, name, engine, *args, **kwargs):
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

    def _load_dump(self, handler, engine, batch_idx, outputs):
        name = self._get_filename(engine, batch_idx)
        return torch.load(f'{engine.handler._dir}/{name}')

    def add_dump(self, name, dir):
        """Add an engine to compare variables.

        Args:
            name (str):
                The name of dump.
            dir (str):
                The directory that the results are saved to.
        """
        with open(f'{dir}/summary') as f:
            summary = json.loads(f.read())

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, *args, **kwargs):
                return ()

        model = DummyModel()
        ppe.to(model, 'cpu')

        # Create a dummy engine
        engine = None
        args = []
        if summary['evaluator']:
            engine = _engine_module.create_evaluator(model)
            args = [[None] * summary['eval_len']]
        if summary['trainer']:
            engine = _engine_module.create_trainer(
                {'main': model},
                {'main': torch.optim.SGD(model.parameters(), lr=0.01)},
                summary['max_epochs'],
                evaluator=engine,
            )
            args = [[None] * summary['train_len']] + args

        _overwrite_handler(
            engine, name, self._load_dump, self._compare_targets,
            self._trigger, dir=dir)

        self._engines[name] = engine, args, {}

    def _dump_targets(self, handler, engine, batch_idx, target):
        name = self._get_filename(engine, batch_idx)
        torch.save(target, f'{engine.handler._dir}/{name}')

    def dump(self, engine, dir, *args, **kwargs):
        """Add an engine to compare variables.

        Args:
            engine (Trainer or Evaluator):
                An engine to compare variables.
            dir (str):
                Name of the directory that the results are saved to.
            *args and **kwargs:
                Arguments passed to ``engine.run``.
        """
        _overwrite_handler(
            engine, None, self._get_target, self._dump_targets, self._trigger, dir=dir)

        engine.run(*args, **kwargs)
        engine.handler = engine.handler._handler
        summary = {
            'evaluator': (isinstance(engine, _evaluator.Evaluator)
                          or getattr(engine, 'evaluator', None) is not None),
            'trainer': isinstance(engine, _trainer.Trainer),
            'train_len': getattr(engine, '_train_len', None),
            'eval_len': engine._eval_len,
        }
        if isinstance(engine,_trainer.Trainer):
            summary['max_epochs'] = engine._manager.max_epochs
        with open(f'{dir}/summary', 'w') as f:
            f.write(json.dumps(summary))

    def _run_engine(self, engine, args, kwargs):
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

    def compare(self):
        """Compares outputs.
        """
        n_workers = len(self._engines)
        self._barrier = threading.Barrier(n_workers)
        self._semaphore = threading.Semaphore(
            n_workers if self._concurrency is None else self._concurrency)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for _, (engine, args, kwargs) in self._engines.items():
                futures.append(executor.submit(self._run_engine, engine, args, kwargs))
            for future in concurrent.futures.as_completed(futures):
                future.result()


def intermediate_value(name: str, value: torch.Tensor) -> None:
    if not hasattr(_thread_local, 'handler'):
        return
    _thread_local.handler._add_intermediate_value(name, value)
