import contextlib
import queue
from typing import (
    Any, Callable, Generator, Iterable, Mapping, Optional, Sequence,
    Union, TYPE_CHECKING,
)

import torch

from pytorch_pfn_extras import reporting
from pytorch_pfn_extras.training.extensions import evaluator

from pytorch_pfn_extras.training.metrics import Batch as DictBatch

if TYPE_CHECKING:
    from pytorch_pfn_extras.handler import BaseHandler
    from pytorch_pfn_extras.training.metrics import MetricType
    from pytorch_pfn_extras.reporting import Observation


@contextlib.contextmanager
def _progress_bar(
        name: str,
        required: bool,
        size: int,
) -> Generator[Callable[[int], None], None, None]:
    if required:
        progress = evaluator.IterationStatus(size)
        pbar = evaluator._IteratorProgressBar(name, progress)

        def update(i: int) -> None:
            progress.current_position = i
            pbar.update()
        yield update

        pbar.close()
    else:
        yield lambda i: None


class Evaluator:
    def __init__(
            self,
            handler: 'BaseHandler',
            models: Union[torch.nn.Module, Mapping[str, torch.nn.Module]],
            *,
            progress_bar: bool = False,
            metrics: Optional[Sequence['MetricType']] = None,
    ):
        super().__init__()

        if not isinstance(models, dict):
            if not isinstance(models, torch.nn.Module):
                raise ValueError(
                    'model must be an instance of dict or toch.nn.Module')
            self.models = {'main': models}
        else:
            self.models = models

        self.handler = handler
        self._progress_bar = progress_bar
        self._reporter = reporting.Reporter()
        self._metrics = [] if metrics is None else metrics
        for name, model in self.models.items():
            self._reporter.add_observer(name, model)
            self._reporter.add_observers(
                name, model.named_modules())

    def _process_metrics(self, ins: DictBatch, outs: DictBatch) -> DictBatch:
        for metric in self._metrics:
            outs.update(metric(ins, outs))
        return outs

    def _complete_step(
            self, idx: int, outs: DictBatch, *, is_deferred: bool = False
    ) -> None:
        c_idx = self._idxs.get()
        # Asure that iterations complete in order
        if c_idx != idx:
            raise RuntimeError(
                'Completed a not expected iteration. '
                '{} was expected but completion of {} happened'.format(
                    c_idx, idx)
            )
        x = self._inputs.get()
        observed = self._observed.get()
        with self._reporter.scope(observed):
            outs = self._process_metrics(x, outs)
            self.handler.eval_post_step(self, idx, x, outs)
        self._summary.add(observed)
        self._update(idx)
        # On the last iteration, close the progress bar
        if self._idxs.qsize() == 0:
            self._pbar.__exit__(None, None, None)

    def run(
            self,
            loader: Iterable[Any],
            *,
            eval_len: Optional[int] = None
    ) -> None:
        """Executes the evaluation loop.

        Args:
            loader (torch.utils.data.DataLoader):
                A data loader for evaluation.
            eval_len (int, optional):
                The number of iterations per one evaluation epoch.
        """
        # Note: setup_manager is done by the Trainer.
        self._idxs: 'queue.Queue[int]' = queue.Queue()
        self._inputs: 'queue.Queue[DictBatch]' = queue.Queue()
        self._observed: 'queue.Queue[Observation]' = queue.Queue()

        if eval_len is None:
            eval_len = len(loader)  # type: ignore[arg-type]
        self._eval_len = eval_len

        self._summary = reporting.DictSummary()
        observation: Observation = {}
        self.handler.eval_loop_begin(self)
        self._pbar = _progress_bar('validation', self._progress_bar, eval_len)
        self._update = self._pbar.__enter__()
        loader_iter = iter(loader)
        with torch.no_grad():  # type: ignore[no-untyped-call]
            for idx in range(eval_len):
                try:
                    x = next(loader_iter)
                except StopIteration:
                    break
                self._idxs.put(idx)
                self._inputs.put(x)
                self._observed.put(observation)
                with self._reporter.scope(observation):
                    self.handler.eval_step(
                        self, idx, x, self._complete_step)
                # Some of the DataLoaders might need an explicit break
                # since they could start cycling on their data
                if (idx + 1) == eval_len:
                    break
        # This will report to the trainer main reporter
        self.handler.eval_loop_end(self)
        reporting.report(self._summary.compute_mean())


# For backward compatibility
_Evaluator = Evaluator
