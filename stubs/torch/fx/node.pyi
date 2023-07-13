# flake8: noqa
from typing import Any, Callable, Dict, List, Optional, Tuple

class Node:
    op: str
    name: str
    target: Any
    meta: Any
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    ...
