import pkg_resources
from packaging.version import Version


def requires(version: str, package: str = 'torch') -> bool:
    pkg_ver = pkg_resources.get_distribution(package).version
    return Version(pkg_ver.split("+")[0].split("-")[0]) >= Version(version)
