import pytest
import torch

import pytorch_pfn_extras as ppe


def test_stream():
    cupy = pytest.importorskip('cupy')

    assert 0 == cupy.cuda.get_current_stream().ptr
    assert 0 == torch.cuda.current_stream().cuda_stream

    # Use the default stream.
    cupy.arange(10)
    torch.arange(10)

    # Use the custom stream.
    stream = torch.cuda.Stream()
    with ppe.cuda.stream(stream):
        cupy.arange(10)
        torch.arange(10)
        assert cupy.cuda.get_current_stream().ptr == stream.cuda_stream

    assert 0 == cupy.cuda.get_current_stream().ptr
    assert 0 == torch.cuda.current_stream().cuda_stream


def test_stream_no_cupy():
    stream = torch.cuda.Stream()
    with ppe.cuda.stream(stream):
        assert torch.cuda.current_stream().cuda_stream == stream.cuda_stream


def test_stream_none():
    assert 0 == torch.cuda.current_stream().cuda_stream
    with ppe.cuda.stream(None):
        assert 0 == torch.cuda.current_stream().cuda_stream


class TestMemoryPool:
    @pytest.fixture
    def cupy(self):
        cupy = pytest.importorskip('cupy')
        mempool = cupy.get_default_memory_pool()
        yield cupy
        mempool.free_all_blocks()
        cupy.cuda.set_allocator(mempool.malloc)

    def test_use_default_mempool(self, cupy):
        # disable mempool
        mempool = cupy.get_default_memory_pool()
        used_bytes = mempool.used_bytes()
        cupy.cuda.set_allocator(None)

        arr1 = cupy.zeros(10)
        assert used_bytes == mempool.used_bytes()

        ppe.cuda.use_default_mempool_in_cupy()

        arr2 = cupy.zeros(10)
        assert used_bytes < mempool.used_bytes()

        del arr1
        del arr2

    def test_use_torch_mempool(self, cupy):
        mempool = cupy.get_default_memory_pool()
        used_bytes = mempool.used_bytes()

        arr1 = cupy.zeros(10)
        assert used_bytes < mempool.used_bytes()

        used_bytes = mempool.used_bytes()
        ppe.cuda.use_torch_mempool_in_cupy()

        arr2 = cupy.zeros(10)
        assert used_bytes == mempool.used_bytes()

        del arr1
        del arr2

    def test_use_torch_mempool_stream(self, cupy):
        ppe.cuda.use_torch_mempool_in_cupy()
        stream = torch.cuda.Stream()
        with ppe.cuda.stream(stream):
            arr1 = torch.arange(10)
            arr2 = cupy.arange(10)
            assert (arr1.numpy() == arr2.get()).all()

        del arr1
        del arr2

    def test_use_torch_mempool_stream_mismatch(self, cupy):
        ppe.cuda.use_torch_mempool_in_cupy()
        stream = cupy.cuda.Stream()
        try:
            stream.use()
            with pytest.raises(
                    RuntimeError, match='pytorch_pfn_extras.cuda.stream'):
                arr = cupy.arange(10)
                del arr
        finally:
            cupy.cuda.Stream.null.use()
