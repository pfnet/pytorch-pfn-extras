import re
from typing import List


def _get_ignite_version(version: str) -> List[int]:
    # We compare up to the minor version (first two digits).
    # This is because it is highly unlikely that these numbers
    # will contain other character than digits.

    # Ignite versioning system is not explicitly documented.
    # However, it seems to be using semver, so the
    # major and minor ids can be only integers.
    # Some examples of versions are:
    # 0.1.0, 0.1.1, 0.3.0.dev20191007, 0.3.0.
    version_regexp = r'^[0-9]+\.[0-9]+\.[0-9]+(\.[0-9a-zA-Z]+)?$'
    if re.search(version_regexp, version):
        return [int(x) for x in version.split('.')[:2]]
    raise ValueError('Invalid version format')
