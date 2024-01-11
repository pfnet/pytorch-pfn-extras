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


class AverageSummary(SummaryBase):
    def __init__(self) -> None:
        self._x: Scalar = 0.0
        self._n: Scalar = 0
        super().__init__()

    def add(self, value: Value, weight: Scalar = 1) -> None:
        if callable(value):
            self._deferred.append((value, weight))
            return
        m = self._n / (self._n + weight)
        self._x = self._x * m + value / weight * (1 - m)
        self._n += weight

    def state_dict(self) -> Dict[str, Any]:
        self._add_deferred_values()
        state = {}
        try:
            # Save the stats as python scalars in order to avoid
            # different device errors when loading them back
            state = {
                "_x": float(self._x),
                "_n": int(self._n),
            }
        except KeyError:
            warnings.warn("The previous statistics are not saved.")
        return state

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        self._add_deferred_values()
        self._x = float(nograd(to_load["_x"]))
        self._n = int(nograd(to_load["_n"]))

    def compute_average(self) -> Scalar:
        self._add_deferred_values()
        return self._x

    def compute_accumulate(self) -> Scalar:
        return self.compute_average()

    def __add__(self, other: AverageSummary) -> AverageSummary:
        s = AverageSummary()
        m = self._n / (self._n + other._n)
        s._x = self._x * m + other._x * (1 - m)
        s._n = self._n + other._n
        s._deferred = self._deferred + other._deferred
        return s
