# mypy: ignore-errors

from torch.nn.parallel import DistributedDataParallel

from pytorch_pfn_extras.nn.parallel import (
    DistributedDataParallel as PpeDistributedDataParallel,
)


def default_transform_model(n, x):
    if isinstance(x, (DistributedDataParallel, PpeDistributedDataParallel)):
        return x.module
    return x
