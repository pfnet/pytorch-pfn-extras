try:
    import cupy  # NOQA
    _cupy_import_error = None
except Exception as e:
    from pytorch_pfn_extras._stub import _cupy as cupy  # NOQA
    _cupy_import_error = e


def ensure_cupy() -> None:
    if _cupy_import_error is not None:
        raise RuntimeError(
            f'CuPy is not available. Reason:\n{_cupy_import_error}')


def is_cupy_available() -> bool:
    return _cupy_import_error is None


try:
    import ignite  # NOQA
    _ignite_import_error = None
except Exception as e:
    from pytorch_pfn_extras._stub import _ignite as cupy  # NOQA
    _ignite_import_error = e


def ensure_ignite() -> None:
    if _ignite_import_error is not None:
        raise RuntimeError(
            f'Ignite is not available. Reason:\n{_cupy_import_error}')
