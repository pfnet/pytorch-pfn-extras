import contextlib

import torch

from pytorch_pfn_extras._cupy import cupy
from pytorch_pfn_extras._cupy import is_available, ensure_cupy


_allocator = None


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
        if is_available():
            cupy_stream = cupy.cuda.ExternalStream(stream.cuda_stream)
            with cupy_stream:
                yield
        else:
            yield


def use_default_mempool_in_cupy():
    """Use the default memory pool in CuPy."""
    ensure_cupy()
    cupy.cuda.set_allocator(cupy.get_default_memory_pool().malloc)


def use_torch_mempool_in_cupy():
    """Use the PyTorch memory pool in CuPy.

    If you want to use PyTorch's memory pool and non-default CUDA streams,
    streams must be created and managed using PyTorch (using
    `torch.cuda.Stream()` and `pytorch_pfn_extras.cuda.stream(stream)`).
    """
    global _allocator

    ensure_cupy()
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
