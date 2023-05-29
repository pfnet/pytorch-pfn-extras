from pytorch_pfn_extras.training._trigger_util import Trigger  # NOQA
from pytorch_pfn_extras.training._trigger_util import TriggerFunc  # NOQA
from pytorch_pfn_extras.training._trigger_util import TriggerLike  # NOQA
from pytorch_pfn_extras.training._trigger_util import get_trigger  # NOQA
from pytorch_pfn_extras.training._trigger_util import (  # NOQA
    _never_fire_trigger,
)

# For backward compatibility
from pytorch_pfn_extras.training.triggers.interval_trigger import (  # NOQA
    IntervalTrigger,
)
