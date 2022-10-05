from typing import List, Tuple
import torch


def find_inplace(g: torch._C.Graph) -> Tuple[torch._C.Graph, List[torch._C.Node]]:
    g = g.copy()
    torch._C._jit_pass_inline(g)
    nodes = []
    for n in g.nodes():
        if n.kind().endswith('_'):
            nodes.append(n)
    return g, nodes
