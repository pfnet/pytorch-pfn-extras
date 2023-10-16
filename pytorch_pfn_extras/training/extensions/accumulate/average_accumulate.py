from pytorch_pfn_extras.training.extensions.accumulate._accumulate_base import (
    AccumulateBase,
)
from pytorch_pfn_extras.training.extensions.accumulate._summary import (
    AverageSummary,
    SummaryBase,
)

from ._accumulate_utils import all_gather_object


class AverageAccumulate(AccumulateBase):
    @property
    def _summary(self) -> SummaryBase:
        return self._average_summary

    def _init_summary(self) -> None:
        self._average_summary = AverageSummary()

    def _all_reduce_summaries(self) -> SummaryBase:
        summaries = all_gather_object(self._average_summary)
        all_reduced_summary = sum(filter(None, summaries), AverageSummary())
        return all_reduced_summary
