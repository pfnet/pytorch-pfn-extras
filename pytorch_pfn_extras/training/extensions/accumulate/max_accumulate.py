from pytorch_pfn_extras.training.extensions.accumulate._accumulate_base import (
    AccumulateBase,
)
from pytorch_pfn_extras.training.extensions.accumulate._summary import (
    MaxSummary,
    SummaryBase,
)

from ._accumulate_utils import all_gather_object


class MaxAccumulate(AccumulateBase):
    @property
    def _summary(self) -> SummaryBase:
        return self._max_summary

    def _init_summary(self) -> None:
        self._max_summary = MaxSummary()

    def _all_reduce_summaries(self) -> SummaryBase:
        summaries = all_gather_object(self._max_summary)
        all_reduced_summary = sum(filter(None, summaries), MaxSummary())
        return all_reduced_summary
