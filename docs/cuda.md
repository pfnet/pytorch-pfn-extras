# CUDA (CuPy Interoperability)

* `pytorch_pfn_extras.cuda.stream(stream)`
    * Context-manager that selects a given stream.
      This context manager also changes the CuPy's default stream if CuPy is available. When CuPy is not available, the functionality is the same as the PyTorch's counterpart, `torch.cuda.stream()`.

* `pytorch_pfn_extras.cuda.use_torch_mempool_in_cupy()`
    * Use PyTorch's memory pool in CuPy.
      If you want to use PyTorch's memory pool and non-default CUDA streams, streams must be created and managed using PyTorch (using `torch.cuda.Stream()` and `pytorch_pfn_extras.cuda.stream(stream)`).

* `pytorch_pfn_extras.cuda.use_default_mempool_in_cupy()`
    * Use CuPy's default memory pool in CuPy.
