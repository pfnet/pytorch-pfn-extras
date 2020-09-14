import contextlib

import torch

try:
    import cupy
    _cupy_import_error = None
except Exception as e:
    _cupy_import_error = e


_allocator = None


def _ensure_cupy():
    if _cupy_import_error is not None:
        raise RuntimeError(
            'CuPy is not available. Reason: \n{}'.format(_cupy_import_error))


@contextlib.contextmanager
def stream(stream):
    """Context-manager that selects a given stream.

    This context manager also changes the CuPy's default stream if CuPy
    is available. When CuPy is not available, the functionality is the same
    as the PyTorch's counterpart, `torch.cuda.stream()`.
    """

    if stream is None:
        yield
        return

    with torch.cuda.stream(stream):
        if _cupy_import_error is None:
            cupy_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
            with cupy_stream:
                yield
        else:
            yield


def use_default_mempool_in_cupy():
    """Use the default memory pool in CuPy."""
    _ensure_cupy()
    cupy.cuda.set_allocator(cupy.get_default_memory_pool().malloc)


def use_torch_mempool_in_cupy():
    """Use the PyTorch memory pool in CuPy.

    If you want to use PyTorch's memory pool and non-default CUDA streams,
    streams must be created and managed using PyTorch (using
    `torch.cuda.Stream()` and `pytorch_pfn_extras.cuda.stream(stream)`).
    """
    global _allocator

    _ensure_cupy()
    _allocator = cupy.cuda.memory.PythonFunctionAllocator(
        _torch_alloc, _torch_free)
    cupy.cuda.set_allocator(_allocator.malloc)


def _torch_alloc(size, device_id):
    torch_stream_ptr = torch.cuda.current_stream().cuda_stream
    cupy_stream_ptr = cupy.cuda.get_current_stream().ptr
    if torch_stream_ptr != cupy_stream_ptr:
        raise RuntimeError(
            'The current stream set in PyTorch and CuPy must be same.'
            ' Use `pytorch_pfn_extras.cuda.stream` instead of'
            ' `torch.cuda.stream`.')
    return torch.cuda.caching_allocator_alloc(
            size, device_id, torch_stream_ptr)


def _torch_free(mem_ptr, device_id):
    torch.cuda.caching_allocator_delete(mem_ptr)
