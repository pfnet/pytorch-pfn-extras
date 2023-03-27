from typing import Any, Callable, List, Tuple
import torch


# Run jit pass with post lint
def run_jit_pass(p: Callable, g: torch._C.Graph, *args: Any, **kwargs: Any) -> None:
    p(g, *args, **kwargs)
    torch._C._jit_pass_lint(g)


def find_inplace(g: torch._C.Graph) -> Tuple[torch._C.Graph, List[torch._C.Node]]:
    g = g.copy()
    run_jit_pass(torch._C._jit_pass_inline, g)
    nodes = []
    for n in g.nodes():
        if n.kind().endswith('_'):
            nodes.append(n)
    return g, nodes
