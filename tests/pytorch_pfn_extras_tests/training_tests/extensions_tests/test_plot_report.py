import warnings

import pytest

from pytorch_pfn_extras.training import extensions


@pytest.fixture(scope="module")
def matplotlib_or_none():
    try:
        import matplotlib

        return matplotlib
    except ImportError:
        return None


@pytest.fixture(scope="module")
def matplotlib(matplotlib_or_none):
    if matplotlib_or_none is None:
        pytest.skip("matplotlib is not installed")
    return matplotlib_or_none


def test_available(matplotlib_or_none):
    if matplotlib_or_none is not None:
        assert extensions.PlotReport.available() is True
    else:
        # It shows warning only when matplotlib is not available
        with pytest.warns(UserWarning):
            assert extensions.PlotReport.available() is False


# TODO(kataoka): lazy import does not seem to be required with matplotlib v3
def test_lazy_import(matplotlib):
    # matplotlib.pyplot should be lazily imported because matplotlib.use
    # has to be called earlier.

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        matplotlib.use("Agg")
        # Test again with a different backend, because the above does not
        # generate a warning if matplotlib.use('Agg') is called and then
        # matplotlib.pyplot is imported.
        matplotlib.use("PS")
