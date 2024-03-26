from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch.distributed
from pytorch_pfn_extras import reporting
from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training._manager_protocol import (
    ExtensionsManagerProtocol,
)
from pytorch_pfn_extras.training.extensions.accumulate._summary import (
    SummaryBase,
)
from pytorch_pfn_extras.training.trigger import TriggerLike, get_trigger


class AccumulateBase(ABC, extension.Extension):
    priority = extension.PRIORITY_EDITOR

    def __init__(
        self,
        conversion_key_pair: Tuple[str, str],
        trigger: TriggerLike = (1, "epoch"),
        distributed: bool = False,
    ) -> None:
        self._conversion_key_pair = conversion_key_pair
        self._trigger = get_trigger(trigger=trigger)
        self._distributed = distributed
        if not torch.distributed.is_initialized() and self._distributed:  # type: ignore[no-untyped-call]
            raise RuntimeError("PyTorch distributed module is not initialized.")

        self._init_summary()

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        observation = manager.observation
        src_key, dst_key = self._conversion_key_pair
        self._summary.add(observation[src_key])

        if self._trigger(manager=manager):
            if self._distributed:
                summary = self._all_reduce_summaries()
            else:
                summary = self._summary
            reporting.report({dst_key: summary.compute_accumulate()})
            self._init_summary()

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        if hasattr(self._trigger, "state_dict"):
            state["_trigger"] = self._trigger.state_dict()
        state["_summary"] = self._summary.state_dict()
        return state

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        if hasattr(self._trigger, "load_state_dict"):
            self._trigger.load_state_dict(to_load["_trigger"])
        self._summary.load_state_dict(to_load["_summary"])

    @property
    @abstractmethod
    def _summary(self) -> SummaryBase: ...

    @abstractmethod
    def _init_summary(self) -> None: ...

    @abstractmethod
    def _all_reduce_summaries(self) -> SummaryBase: ...
