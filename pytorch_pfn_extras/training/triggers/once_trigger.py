from typing import Any, Dict

from pytorch_pfn_extras.training import trigger
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol


class OnceTrigger(trigger.Trigger):

    """Trigger based on the starting point of the iteration.

    This trigger accepts only once at starting point of the iteration. There
    are two ways to specify the starting point: only starting point in whole
    iteration or called again when training resumed.

    Args:
        call_on_resume (bool): Whether the extension is called again or not
            when restored from a snapshot. It is set to ``False`` by default.

    Attributes:
        finished (bool): Flag that indicates whether or not this trigger will
        fire in the future. This flag is used to determine if the extension
        should be initialized after resume.

    """

    def __init__(self, call_on_resume: bool = False) -> None:
        self._flag_first = True
        self._flag_resumed = call_on_resume

    @property
    def finished(self) -> bool:
        return not (self._flag_first or self._flag_resumed)

    def __call__(self, manager: ExtensionsManagerProtocol) -> bool:
        fire = not self.finished
        self._flag_resumed = False
        self._flag_first = False
        return fire

    def state_dict(self) -> Dict[str, Any]:
        state = {'_flag_first': self._flag_first}
        return state

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        self._flag_first = to_load['_flag_first']

    def may_fire(self, iteration: int, epoch_length: int) -> bool:
        return not (self._flag_first or self._flag_resumed)
