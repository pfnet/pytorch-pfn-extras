from pytorch_pfn_extras.handler._code_block import CodeBlock, update_parameters, forward  # NOQA
from pytorch_pfn_extras.handler._handler import BaseHandler, Handler  # NOQA
from pytorch_pfn_extras.handler._logic import BaseLogic, Logic, CodeBlockLogic, ClousureLogic  # NOQA

# Deprecated, only imported for backward compatibility
from pytorch_pfn_extras.handler._logic import torch_autocast  # NOQA
