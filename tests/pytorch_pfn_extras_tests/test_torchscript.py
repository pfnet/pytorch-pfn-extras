import pytorch_pfn_extras.torchscript as ts
import torch


def test_find_inplace():
    def f(v: torch.Tensor) -> None:
        v += torch.ones((1, 2, 3))

    def g(v: torch.Tensor):
        f(v)

    s = torch.jit.script(g)

    new_g, inplace_nodes = ts.find_inplace(s.graph)
    assert len(inplace_nodes) == 1


def test_find_inplace_not_found():
    def f(v: torch.Tensor) -> torch.Tensor:
        return torch.ones((1, 2, 3))

    s = torch.jit.script(f)

    new_g, inplace_nodes = ts.find_inplace(s.graph)
    assert len(inplace_nodes) == 0
