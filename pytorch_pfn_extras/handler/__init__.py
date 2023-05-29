from pytorch_pfn_extras.handler._code_block import (  # NOQA
    CodeBlock,
    forward,
    update_parameters,
)
from pytorch_pfn_extras.handler._handler import BaseHandler, Handler  # NOQA

# Deprecated, only imported for backward compatibility
from pytorch_pfn_extras.handler._logic import torch_autocast  # NOQA
from pytorch_pfn_extras.handler._logic import (  # NOQA
    BaseLogic,
    ClousureLogic,
    CodeBlockLogic,
    Logic,
)
