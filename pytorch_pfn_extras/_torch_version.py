from packaging.version import Version


def requires(version: str, package: str = "torch") -> bool:
    if package == "torch":
        import torch as module
    elif package == "onnx":
        import onnx as module
    else:
        raise ValueError(f"Unsupported package: {package}")
    pkg_ver = module.__version__
    return Version(pkg_ver.split("+")[0].split("-")[0]) >= Version(version)
