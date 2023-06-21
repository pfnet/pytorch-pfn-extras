# flake8: noqa
from typing import Any, Dict

import torch

from .graph import Graph

class GraphModule:
    graph: Graph

    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: Graph,
        class_name: str = "GraphModule",
    ): ...
    ...
