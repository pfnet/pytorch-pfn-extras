import importlib.metadata

from packaging.version import Version


def requires(version: str, package: str = "torch") -> bool:
    pkg_ver = importlib.metadata.version(package)
    return Version(pkg_ver.split("+")[0].split("-")[0]) >= Version(version)
