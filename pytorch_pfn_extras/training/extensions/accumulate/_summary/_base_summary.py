from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

from pytorch_pfn_extras.reporting import Scalar, Value


class SummaryBase(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._deferred: List[Tuple[Callable[[], float], Scalar]] = []

    @abstractmethod
    def add(self, value: Value, weight: Scalar = 1) -> None: ...

    @abstractmethod
    def compute_accumulate(self) -> Scalar: ...

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]: ...

    @abstractmethod
    def load_state_dict(self, to_load: Dict[str, Any]) -> None: ...

    def _add_deferred_values(self) -> None:
        for fn, weight in self._deferred:
            value = fn()
            self.add(value=value, weight=weight)
        self._deferred.clear()
