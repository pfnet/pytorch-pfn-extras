from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
)

from pytorch_pfn_extras.training import trigger as trigger_module
from pytorch_pfn_extras.training._manager_protocol import (
    ExtensionsManagerProtocol,
)

if TYPE_CHECKING:
    from pytorch_pfn_extras.training._trigger_util import TriggerLike


class FunctionTrigger(trigger_module.Trigger):
    def __init__(
        self,
        fn: Callable[..., bool],
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        trigger: "TriggerLike" = (1, "iteration"),
    ) -> None:
        self._fn = fn
        self._args = args or []
        self._kwargs = kwargs or {}
        self._interval_trigger = trigger_module.get_trigger(trigger)

    def __call__(self, manager: ExtensionsManagerProtocol) -> bool:
        if not self._interval_trigger(manager):
            return False

        return self._fn(*self._args, **self._kwargs)

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "interval_trigger": self._interval_trigger.state_dict(),
        }
        return state

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        self._interval_trigger.load_state_dict(to_load["interval_trigger"])

    def may_fire(self, iteration: int, epoch_len: int) -> bool:
        if self._interval_trigger.may_fire(
            iteration=iteration, epoch_len=epoch_len
        ):
            return self._fn(*self._args, **self._kwargs)
        else:
            return False
