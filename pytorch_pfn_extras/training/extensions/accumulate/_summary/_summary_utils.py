import torch
from pytorch_pfn_extras.reporting import Scalar


def nograd(value: Scalar) -> Scalar:
    if isinstance(value, torch.Tensor):
        return value.detach()
    return value
