# mypy: ignore-errors

import re
import threading
import concurrent.futures

import torch.testing

from pytorch_pfn_extras import handler as _handler_module
from pytorch_pfn_extras.training import _trainer
from pytorch_pfn_extras.training import _evaluator


class _ComparableHandler(_handler_module.BaseHandler):
    def __init__(self, handler, name, save_outs_cb):
        self._handler = handler
        self._save_outs_cb = save_outs_cb
        self.name = name
        self.iteration = 0

    def convert_batch(self, args):
        return self._handler.convert_batch(args)

    def train_setup(self, trainer, loader):
        return self._handler.train_setup(trainer, loader)

    def train_epoch_begin(self, trainer, loader):
        return self._handler.train_epoch_begin(trainer, loader)

    def train_epoch_end(self, trainer):
        return self._handler.train_epoch_end(trainer)

    def train_validation_begin(self, evaluator):
        return self._handler.train_validation_begin(evaluator)

    def train_validation_end(self, trainer, evaluator):
        return self._handler.train_validation_end(trainer, evaluator)

    def train_step(self, trainer, batch_idx, batch, complete_fn):
        return self._handler.train_step(trainer, batch_idx, batch, complete_fn)

    def train_post_step(self, trainer, batch_idx, batch, outputs):
        self._handler.train_post_step(trainer, batch_idx, batch, outputs)
        self.iteration += 1
        return self._save_outs_cb(self, trainer.models, batch_idx, outputs)

    def eval_loop_begin(self, evaluator):
        return self._handler.eval_loop_begin(evaluator)

    def eval_step(self, evaluator, batch_idx, batch, complete_fn):
        return self._handler.eval_step(
            evaluator, batch_idx, batch, complete_fn)

    def eval_loop_end(self, evaluator):
        return self._handler.eval_loop_end(evaluator)

    def eval_post_step(self, evaluator, batch_idx, batch, outputs):
        self._handler.eval_post_step(evaluator, batch_idx, batch, outputs)
        self.iteration += 1
        return self._save_outs_cb(self, evaluator.models, batch_idx, outputs)


def get_default_comparer(rtol=1e-07, atol=0, equal_nan=True, msg=None):
    """Creates default comparer function.

    The created function will compare the outputs by using
    `torch.testing.assert_allclose` with specified options.

    Args:
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.
        equal_nan (bool): If ``True``, NaNs will be ignored.
        msg (str): Error message to be printed in case of failure.
    """
    def compare_fn(backend1, backend2, name, val1, val2):
        err_msg = msg or f" comparing {backend1} and {backend2} in {name}"
        torch.testing.assert_allclose(
            # TODO select the device where
            # the tensors will be compared?
            val1.cpu().detach(),
            val2.cpu().detach(),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            msg=err_msg,
        )
    return compare_fn


_default_comparer = get_default_comparer()


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
        # engines must be a dict
        for name, engine in engines.items():
            engine.handler = _ComparableHandler(
                engine.handler, name, self.compare_targets
            )

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

    def _add_target(self, handle, models, outputs):
        raise NotImplementedError('Comparers must override _add_target')

    def compare_targets(self, handle, models, batch_idx, outputs):
        if (self.n_iters is None) or (handle.iteration % self.n_iters == 0):
            # Save the outputs of this iteration
            with self.report_lock:
                self._add_target(handle, models, outputs)
                if len(self.targets.keys()) == len(self.engines.keys()):
                    # all outputs have been filled, lets compare and reset
                    self._compare_targets()
                    self.targets = {}
                self._assert_incompatible_trigger(not self._finalized)
            # Excplicitly synchronize
            self._semaphore.release()
            self.barrier.wait()
            self._semaphore.acquire()

    def _compare_targets(self):
        names = list(self.targets.keys())
        for i, name in enumerate(names):
            for target in self.targets[name]:
                for j in range(i + 1, len(names)):
                    to_compare = names[j]
                    target_1 = self.targets[name][target]
                    target_2 = self.targets[to_compare][target]
                    self.compare_fn(
                        name, to_compare, target, target_1, target_2)


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

    def _add_target(self, handle, models, outputs):
        keys = (
            self.to_compare_keys
            if self.to_compare_keys is not None
            else outputs.keys()
        )
        self.targets[handle.name] = {key: outputs[key] for key in keys}


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

    def _add_target(self, handle, models, outputs):
        sdict = models['main'].state_dict()
        if self._preprocessed_keys is None:
            self._preprocess_keys(sdict)
        self.targets[handle.name] = {
            key: sdict[key] for key in self._preprocessed_keys}
