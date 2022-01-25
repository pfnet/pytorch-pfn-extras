import torch
from packaging import version


torch_version = version.Version(torch.__version__.split("+")[0])


def is_available(ver: str) -> bool:
    return torch_version >= version.Version(ver)
