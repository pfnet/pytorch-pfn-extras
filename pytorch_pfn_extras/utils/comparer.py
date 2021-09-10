# mypy: ignore-errors

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
        return self._save_outs_cb(self, batch_idx, outputs)

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
        return self._save_outs_cb(self, batch_idx, outputs)


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
    def compare_fn(backend1, backend2, out_name, out1, out2):
        err_msg = msg or f" comparing {backend1} and {backend2} in {out_name}"
        torch.testing.assert_allclose(
            # TODO select the device where
            # the tensors will be compared?
            out1.cpu().detach(),
            out2.cpu().detach(),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            msg=err_msg,
        )
    return compare_fn


_default_comparer = get_default_comparer()


class OutputsComparer:
    def __init__(
            self, engines, to_compare_keys=None, *,
            compare_fn=_default_comparer,
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

        Examples:
            >>> trainer_cpu = ppe.engine.create_trainer(
                    model, optimizer, 1, device='cpu')
            >>> trainer_gpu = ppe.engine.create_trainer(
                    model, optimizer, 1, device='cuda:0')
            >>> comp = ppe.utils.comparer.OutputsComparer(
                    {"cpu": trainer_cpu, "gpu": trainer_gpu})
            >>> comp.compare({"cpu": loader, "gpu": loader}])
        """
        e_type = type(next(iter(engines.values())))
        if e_type not in (
            _trainer._Trainer,
            _evaluator._Evaluator,
        ):
            raise ValueError(f"Engine type {e_type} is not supported")
        if not all((isinstance(e, e_type) for e in engines.values())):
            raise ValueError("All the engines must be of the same type")
        # engines must be a dict
        for name, engine in engines.items():
            engine.handler = _ComparableHandler(
                engine.handler, name, self.report_output
            )

        self.engines = engines  # Need to wrap the handle with ours
        # If to_compare_key is None, then we compare all
        self.outputs = {}
        self.to_compare_keys = to_compare_keys
        self.barrier = threading.Barrier(len(engines))
        self.report_lock = threading.Lock()
        self.compare_fn = compare_fn
        self._finalized = False

    def _assert_incompatible_trigger(self, condition):
        if not condition:
            raise ValueError('Engines have different triggers.')

    def report_output(self, handle, batch_idx, outputs):
        if (self.n_iters is None) or (handle.iteration % self.n_iters == 0):
            keys = (
                self.to_compare_keys
                if self.to_compare_keys is not None
                else outputs.keys()
            )

            # Save the outputs of this iteration
            with self.report_lock:
                self.outputs[handle.name] = {key: outputs[key] for key in keys}
                if len(self.outputs.keys()) == len(self.engines.keys()):
                    # all outputs have been filled, lets compare and reset
                    self._compare_outs()
                    self.outputs = {}
                self._assert_incompatible_trigger(not self._finalized)

            # Excplicitly synchronize
            self.barrier.wait()

    def _compare_outs(self):
        names = list(self.outputs.keys())
        for i, name in enumerate(names):
            for out in self.outputs[name]:
                for j in range(i + 1, len(names)):
                    to_compare = names[j]
                    out_1 = self.outputs[name][out]
                    out_2 = self.outputs[to_compare][out]
                    self.compare_fn(name, to_compare, out, out_1, out_2)

    def run_engine(self, engine, loaders):
        try:
            engine.run(loaders)
            with self.report_lock:
                self._finalized = True
                self._assert_incompatible_trigger(len(self.outputs) == 0)
        except Exception:
            self.barrier.abort()
            raise

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
