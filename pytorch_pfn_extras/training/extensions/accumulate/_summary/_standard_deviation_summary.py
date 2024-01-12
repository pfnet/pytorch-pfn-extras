from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy
import torch
from pytorch_pfn_extras.reporting import Scalar, Value
from pytorch_pfn_extras.training.extensions.accumulate._summary._base_summary import (
    SummaryBase,
)
from pytorch_pfn_extras.training.extensions.accumulate._summary._summary_utils import (
    nograd,
)


class StandardDeviationSummary(SummaryBase):
    def __init__(self) -> None:
        self._x: Scalar = 0.0
        self._x2: Scalar = 0.0
        self._n: Scalar = 0
        super().__init__()

    def add(self, value: Value, weight: Scalar = 1) -> None:
        if callable(value):
            self._deferred.append((value, weight))
            return
        self._x += weight * value
        self._x2 += weight * value * value
        self._n += weight

    def state_dict(self) -> Dict[str, Any]:
        self._add_deferred_values()
        state = {}
        try:
            # Save the stats as python scalars in order to avoid
            # different device errors when loading them back
            state = {
                "_x": float(self._x),
                "_x2": float(self._x2),
                "_n": float(self._n),
            }
        except KeyError:
            warnings.warn("The previous statistics are not saved.")
        return state

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        self._add_deferred_values()
        self._x = float(nograd(to_load["_x"]))
        self._x2 = float(nograd(to_load["_x2"]))
        self._n = float(nograd(to_load["_n"]))

    def compute_mean(self) -> Scalar:
        self._add_deferred_values()
        x, n = self._x, self._n
        return x / n

    def compute_standard_deviation(self) -> Scalar:
        self._add_deferred_values()
        x, n = self._x, self._n
        mean = x / n
        var = self._x2 / n - mean * mean
        if isinstance(var, torch.Tensor):
            return torch.sqrt(var)
        else:
            return numpy.sqrt(var)

    def compute_accumulate(self) -> Scalar:
        return self.compute_standard_deviation()

    def __add__(
        self, other: StandardDeviationSummary
    ) -> StandardDeviationSummary:
        s = StandardDeviationSummary()
        s._x = self._x + other._x
        s._x2 = self._x2 + other._x2
        s._n = self._n + other._n
        s._deferred = self._deferred + other._deferred
        return s
