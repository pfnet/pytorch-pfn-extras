# flake8: noqa
from typing import Any

from .graph import Graph, Node

class GraphAppendingTracer:
    def __init__(self, graph: Graph): ...

class Proxy:
    def __init__(self, node: Node, tracer: GraphAppendingTracer): ...
