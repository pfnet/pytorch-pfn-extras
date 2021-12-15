import numpy
import torch
import torch.utils.dlpack
from typing import Any, Dict, Union

from pytorch_pfn_extras._cupy import cupy
from pytorch_pfn_extras._cupy import ensure_cupy


_NDArray = Any  # TypeVar("_NDArray", numpy.ndarray, cupy.ndarray)
_NumpyDtype = Any  # numpy.dtype


def from_ndarray(ndarray: _NDArray) -> torch.Tensor:
    """Creates a `torch.Tensor` from a `numpy.ndarray` or `cupy.ndarray`.

    Unlike `torch.from_numpy`, this method may make a copy when needed, e.g.
    when the given `ndarray` contains the negative strides which is not
    supported by PyTorch.
    """
    if isinstance(ndarray, cupy.ndarray):
        pack = _copy_if_negative_strides(ndarray).toDlpack()
        try:
            return torch.utils.dlpack.from_dlpack(pack)
        except Exception as e:
            # TODO(kmaehashi): Remove this workaround once PyTorch is fixed.
            # https://github.com/pytorch/pytorch/pull/56789
            # This mitigates a bug above by deferring the destruction of the
            # capsule so that users can see the exception.
            e._dlpack = pack  # type: ignore
            raise
    elif isinstance(ndarray, numpy.ndarray):
        return torch.from_numpy(_copy_if_negative_strides(ndarray))
    raise TypeError(
        'expected numpy.ndarray or cupy.ndarray '
        f'(got {type(ndarray).__name__})')


def _copy_if_negative_strides(ndarray: _NDArray) -> _NDArray:
    # Torch does not support negative strides, make a copy in that case.
    if any(s < 0 for s in ndarray.strides):
        return ndarray.copy()
    return ndarray


def as_ndarray(tensor: torch.Tensor) -> _NDArray:
    """Creates a `numpy.ndarray` or `cupy.ndarray` from `torch.Tensor`.

    This method returns a tensor as a NumPy or CuPy ndarray depending on where
    the given `tensor` resides in. The `tensor` and the returned `ndarray`
    share the same underlying storage. Changes to the tensor will be reflected
    in the `ndarray` and vice versa. Note that changes made to `ndarray`
    cannot be tracked in the computational graph.
    """
    devtype = tensor.device.type
    if devtype == 'cpu':
        return tensor.detach().numpy()
    elif devtype == 'cuda':
        ensure_cupy()
        if hasattr(cupy, 'from_dlpack'):
            # TODO: Avoid using ``torch.utils.dlpack.to_dlpack``.
            # => return cupy.from_dlpack(tensor)
            # Blocked by PyTorch 1.10 bug
            # (https://github.com/pytorch/pytorch/pull/67618)
            return cupy.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))
        return cupy.fromDlpack(torch.utils.dlpack.to_dlpack(tensor))
    raise ValueError(f'Tensor is on unsupported device: {devtype}')


def get_xp(obj: Union[_NDArray, torch.Tensor]) -> Any:
    """Returns a module of ndarray implementation (`numpy` or `cupy`) for the
    given `obj`.

    The `obj` can be `torch.Tensor`, `torch.device` or NumPy/CuPy `ndarray`.
    """
    if isinstance(obj, torch.Tensor):
        devtype = obj.device.type
    elif isinstance(obj, torch.device):
        devtype = obj.type
    elif isinstance(obj, numpy.ndarray):
        devtype = 'cpu'
    elif isinstance(obj, cupy.ndarray):
        devtype = 'cuda'
    else:
        raise TypeError(
            'expected torch.Tensor, torch.device, numpy.ndarray, '
            f'or cupy.ndarray (got {type(obj).__name__})')

    if devtype == 'cpu':
        return numpy
    elif devtype == 'cuda':
        ensure_cupy()
        return cupy

    raise ValueError(f'unsupported device type: {devtype}')


def as_numpy_dtype(torch_dtype: torch.dtype) -> _NumpyDtype:
    """Returns NumPy dtype for the given PyTorch dtype.

    Args:
        torch_dtype: PyTorch's dtype object.

    Returns:
        NumPy type object.

    """
    numpy_dtype = _torch_dtype_mapping.get(torch_dtype, None)
    if numpy_dtype is None:
        raise TypeError(f'NumPy does not support {torch_dtype} equivalent')
    return numpy_dtype


def from_numpy_dtype(numpy_dtype: _NumpyDtype) -> torch.dtype:
    """Returns PyTorch dtype for the given NumPy dtype.

    Args:
        numpy_dtype: NumPy's dtype object.

    Returns:
        PyTorch type object.

    """
    torch_dtype = _numpy_dtype_mapping.get(numpy_dtype, None)
    if torch_dtype is None:
        raise TypeError(f'PyTorch does not support {numpy_dtype} equivalent')
    return torch_dtype


_torch_dtype_mapping: Dict[torch.dtype, _NumpyDtype] = {
    # https://pytorch.org/docs/stable/tensors.html
    # https://numpy.org/doc/stable/user/basics.types.html

    torch.float32: numpy.dtype('float32'),
    torch.float64: numpy.dtype('float64'),
    torch.float16: numpy.dtype('float16'),
    # unsupported: torch.bfloat16
    # unsupported: torch.complex32
    torch.complex64: numpy.dtype('complex64'),
    torch.complex128: numpy.dtype('complex128'),
    torch.uint8: numpy.dtype('uint8'),
    torch.int8: numpy.dtype('int8'),
    torch.int16: numpy.dtype('int16'),
    torch.int32: numpy.dtype('int32'),
    torch.int64: numpy.dtype('int64'),
    torch.bool: numpy.dtype('bool'),
}

_numpy_dtype_mapping: Dict[_NumpyDtype, torch.dtype] = {
    v: k for k, v in _torch_dtype_mapping.items()
}
