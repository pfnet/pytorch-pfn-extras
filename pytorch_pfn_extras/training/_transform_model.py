import typing

import torch
from torch.nn.parallel import DistributedDataParallel

from pytorch_pfn_extras.nn.parallel import (
    DistributedDataParallel as PpeDistributedDataParallel,
)


_TransformModel = typing.Callable[[str, torch.nn.Module], torch.nn.Module]


def default_transform_model(n: str, x: torch.nn.Module) -> torch.nn.Module:
    if isinstance(x, (DistributedDataParallel, PpeDistributedDataParallel)):
        return x.module
    return x
