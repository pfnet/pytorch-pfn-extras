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


class MaxSummary(SummaryBase):
    def __init__(self) -> None:
        self._max_x: Scalar = -float("inf")
        super().__init__()

    def add(self, value: Value, weight: Scalar = 1) -> None:
        if callable(value):
            self._deferred.append((value, weight))
            return
        self._max_x = self._max_x if self._max_x > value else value

    def state_dict(self) -> Dict[str, Any]:
        self._add_deferred_values()
        state = {}
        try:
            # Save the stats as python scalars in order to avoid
            # different device errors when loading them back
            state = {
                "_max_x": float(self._max_x),
            }
        except KeyError:
            warnings.warn("The previous statistics are not saved.")
        return state

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        self._add_deferred_values()
        self._max_x = float(nograd(to_load["_max_x"]))

    def compute_max(self) -> Scalar:
        self._add_deferred_values()
        return self._max_x

    def compute_accumulate(self) -> Scalar:
        return self.compute_max()

    def __add__(self, other: MaxSummary) -> MaxSummary:
        s = MaxSummary()
        s._max_x = self._max_x if self._max_x > other._max_x else other._max_x
        s._deferred = self._deferred + other._deferred
        return s
