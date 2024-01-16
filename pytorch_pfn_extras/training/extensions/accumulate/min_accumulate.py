from pytorch_pfn_extras.training.extensions.accumulate._accumulate_base import (
    AccumulateBase,
)
from pytorch_pfn_extras.training.extensions.accumulate._summary import (
    MinSummary,
    SummaryBase,
)

from ._accumulate_utils import all_gather_object


class MinAccumulate(AccumulateBase):
    @property
    def _summary(self) -> SummaryBase:
        return self._min_summary

    def _init_summary(self) -> None:
        self._min_summary = MinSummary()

    def _all_reduce_summaries(self) -> SummaryBase:
        summaries = all_gather_object(self._min_summary)
        all_reduced_summary = sum(filter(None, summaries), MinSummary())
        return all_reduced_summary
