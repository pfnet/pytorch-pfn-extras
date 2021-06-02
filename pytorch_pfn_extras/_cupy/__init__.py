try:
    import cupy  # NOQA
    _cupy_import_error = None
except Exception as e:
    from pytorch_pfn_extras._cupy import _cupy_stub as cupy  # NOQA
    _cupy_import_error = e


def ensure_cupy() -> None:
    if _cupy_import_error is not None:
        raise RuntimeError(
            f'CuPy is not available. Reason:\n{_cupy_import_error}')


def is_available() -> bool:
    return _cupy_import_error is None
