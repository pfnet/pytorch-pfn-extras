from __future__ import annotations

import warnings
from typing import Any, Dict

from pytorch_pfn_extras.reporting import Scalar, Value
from pytorch_pfn_extras.training.extensions.accumulate._summary._base_summary import (
    SummaryBase,
)
from pytorch_pfn_extras.training.extensions.accumulate._summary._summary_utils import (
    nograd,
)


class MinSummary(SummaryBase):
    def __init__(self) -> None:
        self._min_x: Scalar = float("inf")
        super().__init__()

    def add(self, value: Value, weight: Scalar = 1) -> None:
        if callable(value):
            self._deferred.append((value, weight))
            return
        self._min_x = self._min_x if self._min_x < value else value

    def state_dict(self) -> Dict[str, Any]:
        self._add_deferred_values()
        state = {}
        try:
            # Save the stats as python scalars in order to avoid
            # different device errors when loading them back
            state = {
                "_min_x": float(self._min_x),
            }
        except KeyError:
            warnings.warn("The previous statistics are not saved.")
        return state

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        self._add_deferred_values()
        self._min_x = float(nograd(to_load["_min_x"]))

    def compute_min(self) -> Scalar:
        self._add_deferred_values()
        return self._min_x

    def compute_accumulate(self) -> Scalar:
        return self.compute_min()

    def __add__(self, other: MinSummary) -> MinSummary:
        s = MinSummary()
        s._min_x = self._min_x if self._min_x < other._min_x else other._min_x
        s._deferred = self._deferred + other._deferred
        return s
