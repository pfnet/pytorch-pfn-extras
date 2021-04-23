try:
    import cupy
    _cupy_import_error = None
except Exception as e:
    cupy = None
    _cupy_import_error = e


def ensure_cupy():
    if cupy is None:
        raise RuntimeError(
            f'CuPy is not available. Reason:\n{_cupy_import_error}')
