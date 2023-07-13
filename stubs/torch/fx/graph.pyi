# flake8: noqa
from typing import Any, Callable, Dict, List, Optional, Tuple

from .node import Node

class Graph:
    nodes: List[Node]

    def call_function(
        self,
        the_function: Callable[..., Any],
        args: Optional[Any] = None,
        kwargs: Optional[Any] = None,
        type_expr: Optional[Any] = None,
    ) -> Node: ...
    def output(self, result: Any, type_expr: Optional[Any] = None) -> Node: ...
    def placeholder(
        self,
        name: str,
        type_expr: Optional[Any] = None,
        default_value: Optional[Any] = None,
    ) -> Node: ...
    def node_copy(
        self, node: Node, arg_transform: Callable[[Node], "Argument"]
    ) -> Node: ...
    def inserting_after(self, n: Optional[Node] = None): ...
    def create_node(
        self,
        op: str,
        target: Any,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> Node: ...
    def erase_node(self, to_erase: Node) -> None: ...
    ...
