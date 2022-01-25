import torch
from packaging import version


torch_version = version.Version(torch.__version__.split("+")[0])


def requires(version: str, *, package='torch') -> bool:
    return torch_version >= version.Version(ver)
