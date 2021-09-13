import pytest
import torch

from pytorch_pfn_extras.nn import EnsureShapeAndDtype


class TestEnsureShapeAndDtype:

    def test_wrong_initialization(self):
        with pytest.raises(ValueError, match='both arguments'):
            EnsureShapeAndDtype(None, None)

    @pytest.mark.parametrize(
        'shape', [(), (1,), (1, 1), (2,), (2, 4), (2, 3, 4)]
    )
    def test_valid_shape(self, shape):
        tensor = torch.zeros(shape)
        module = EnsureShapeAndDtype(shape)
        module(tensor)

    @pytest.mark.parametrize(
        'shape', [(), (1,), (1, 1), (2,), (2, 4), (2, 3, 4)]
    )
    def test_invalid_shape(self, shape):
        tensor = torch.zeros((1, 2, 3))
        module = EnsureShapeAndDtype(shape)
        with pytest.raises(ValueError, match='input shape is'):
            module(tensor)

    @pytest.mark.parametrize('shape_t, shape_c', [
         ((1,), (2,)),
         ((1, 1), (2, 1)),
         ((2, 1), (2, 2)),
         ((2, 4), (1, 4)),
         ((2, 3, 4), (2, 1, 4)),
    ])
    def test_broadcastable_shape(self, shape_t, shape_c):
        tensor = torch.zeros(shape_t)
        module = EnsureShapeAndDtype(shape_c, broadcastable=True)
        module(tensor)

    @pytest.mark.parametrize('shape_t, shape_c', [
         ((3,), (2,)),
         ((2, 3), (2, 2)),
         ((2, 4), (3, 4)),
         ((2, 3, 4), (2, 2, 1)),
    ])
    def test_nonbroadcastable_shape(self, shape_t, shape_c):
        tensor = torch.zeros(shape_t)
        module = EnsureShapeAndDtype(shape_c, broadcastable=True)
        with pytest.raises(ValueError, match='non broadcastable'):
            module(tensor)

    @pytest.mark.parametrize(
        'dtype', [torch.int32, torch.float32, torch.complex64]
    )
    def test_valid_dtypes(self, dtype):
        tensor = torch.zeros(1, dtype=dtype)
        module = EnsureShapeAndDtype(None, dtype)
        module(tensor)

    @pytest.mark.parametrize('dtype_t, dtype_c', [
        (torch.int32, torch.int16),
        (torch.int32, torch.float32),
        (torch.float32, torch.float64),
        (torch.float32, torch.complex64),
    ])
    def test_invalid_dtypes(self, dtype_t, dtype_c):
        tensor = torch.zeros(1, dtype=dtype_t)
        module = EnsureShapeAndDtype(None, dtype_c)
        with pytest.raises(ValueError, match='input dtype'):
            module(tensor)

    @pytest.mark.parametrize('dtype_t, dtype_c', [
        (torch.int32, torch.float32),
        (torch.int32, torch.complex128),
        (torch.int8, torch.float16),
    ])
    def test_dtypes_with_cast(self, dtype_t, dtype_c):
        tensor = torch.zeros(1, dtype=dtype_t)
        module = EnsureShapeAndDtype(None, dtype_c, can_cast=True)
        module(tensor)

    @pytest.mark.parametrize('dtype_t, dtype_c', [
        (torch.complex64, torch.int32),
        (torch.float32, torch.int32),
    ])
    def test_invalid_dtypes_with_cast(self, dtype_t, dtype_c):
        tensor = torch.zeros(1, dtype=dtype_t)
        module = EnsureShapeAndDtype(None, dtype_c, can_cast=True)
        with pytest.raises(ValueError, match='be casted to'):
            module(tensor)

    def test_valid_shape_and_dtype(self):
        shape = (10, 5)
        dtype = torch.float32
        tensor = torch.zeros(shape, dtype=dtype)
        module = EnsureShapeAndDtype(shape, dtype)
        module(tensor)
