from pytorch_pfn_extras.training.extensions.accumulate._accumulate_base import (
    AccumulateBase,
)
from pytorch_pfn_extras.training.extensions.accumulate._summary import (
    StandardDeviationSummary,
    SummaryBase,
)

from ._accumulate_utils import all_gather_object


class StandardDeviationAccumulate(AccumulateBase):
    @property
    def _summary(self) -> SummaryBase:
        return self._standard_deviation_summary

    def _init_summary(self) -> None:
        self._standard_deviation_summary = StandardDeviationSummary()

    def _all_reduce_summaries(self) -> SummaryBase:
        summaries = all_gather_object(self._standard_deviation_summary)
        all_reduced_summary = sum(
            filter(None, summaries), StandardDeviationSummary()
        )
        return all_reduced_summary
