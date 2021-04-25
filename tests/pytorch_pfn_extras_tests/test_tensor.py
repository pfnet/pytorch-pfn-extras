import numpy
import pytest
import torch
import torch.utils.dlpack

import pytorch_pfn_extras as ppe


def test_from_ndarray_numpy():
    np_arr = numpy.arange(24).reshape(2, 3, 4)
    tensor = ppe.from_ndarray(np_arr)
    assert np_arr.ctypes.data == tensor.data_ptr()
    numpy.testing.assert_array_equal(np_arr, tensor.numpy())


def test_from_ndarray_numpy_neg():
    np_arr = numpy.flip(numpy.arange(24).reshape(2, 3, 4))
    tensor = ppe.from_ndarray(np_arr)  # copy
    assert np_arr.ctypes.data != tensor.data_ptr()
    numpy.testing.assert_array_equal(np_arr, tensor.numpy())


def test_from_ndarray_cupy():
    cupy = pytest.importorskip('cupy')
    cp_arr = cupy.arange(24).reshape(2, 3, 4)
    tensor = ppe.from_ndarray(cp_arr)
    assert cp_arr.data.ptr == tensor.data_ptr()
    numpy.testing.assert_array_equal(cp_arr.get(), tensor.cpu().numpy())


def test_from_ndarray_cupy_neg():
    cupy = pytest.importorskip('cupy')
    cp_arr = cupy.flip(cupy.arange(24).reshape(2, 3, 4))
    tensor = ppe.from_ndarray(cp_arr)  # copy
    assert cp_arr.data.ptr != tensor.data_ptr()
    numpy.testing.assert_array_equal(cp_arr.get(), tensor.cpu().numpy())


def test_from_ndarray_invalid_type():
    with pytest.raises(TypeError):
        ppe.from_ndarray([1, 2, 3])


def test_as_ndarray_cpu():
    tensor = torch.arange(24).reshape(2, 3, 4)
    arr = ppe.as_ndarray(tensor)
    assert isinstance(arr, numpy.ndarray)
    assert tensor.data_ptr() == arr.ctypes.data
    numpy.testing.assert_array_equal(tensor.numpy(), arr)


def test_as_ndarray_cupy():
    cupy = pytest.importorskip('cupy')
    tensor = torch.arange(24).reshape(2, 3, 4).cuda()
    arr = ppe.as_ndarray(tensor)
    assert isinstance(arr, cupy.ndarray)
    assert tensor.data_ptr() == arr.data.ptr
    numpy.testing.assert_array_equal(tensor.cpu().numpy(), arr.get())


def test_get_xp_numpy():
    assert ppe.get_xp(torch.ones(4)) is numpy
    assert ppe.get_xp(torch.device('cpu')) is numpy
    assert ppe.get_xp(numpy.ones(4)) is numpy


def test_get_xp_cupy():
    cupy = pytest.importorskip('cupy')
    assert ppe.get_xp(torch.ones(4).cuda()) is cupy
    assert ppe.get_xp(torch.device('cuda:0')) is cupy
    assert ppe.get_xp(cupy.ones(1)) is cupy


def test_get_xp_invalid_type():
    with pytest.raises(TypeError):
        ppe.get_xp([1, 2, 3])


@pytest.mark.parametrize('dtype', [
    'bool',
    'uint8', 'int8', 'int16', 'int32', 'int64',
    'float16', 'float32', 'float64',
    'complex64', 'complex128',
])
def test_torch_numpy_dtype(dtype):
    torch_dtype = getattr(torch, dtype)
    numpy_dtype = numpy.dtype(dtype)
    assert ppe.as_numpy_dtype(torch_dtype) == numpy_dtype
    assert ppe.from_numpy_dtype(numpy_dtype) == torch_dtype


def test_torch_numpy_dtype_unsupported():
    with pytest.raises(TypeError):
        ppe.as_numpy_dtype(torch.bfloat16)
    with pytest.raises(TypeError):
        ppe.from_numpy_dtype(numpy.dtype('object'))
    with pytest.raises(TypeError):
        ppe.from_numpy_dtype(None)
