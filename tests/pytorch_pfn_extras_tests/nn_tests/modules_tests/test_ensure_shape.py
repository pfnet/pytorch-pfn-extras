import pytest
import torch

from pytorch_pfn_extras.nn import EnsureShapeAndDtype, ensure_shape_and_dtype


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
        # Use the function version
        ensure_shape_and_dtype(tensor, shape)

    @pytest.mark.parametrize(
        'shape', [(), (1,), (1, 1), (2,), (2, 4), (2, 3, 4)]
    )
    def test_invalid_shape(self, shape):
        tensor = torch.zeros((1, 2, 3))
        module = EnsureShapeAndDtype(shape)
        with pytest.raises(ValueError, match='input shape is'):
            module(tensor)
        with pytest.raises(ValueError, match='input shape is'):
            ensure_shape_and_dtype(tensor, shape)

    @pytest.mark.parametrize('shape_t, shape_c', [
         ((1,), (2,)),
         ((1, 1), (2, 1)),
         ((1, 1), (2,)),
         ((2, 1), (2, 2)),
         ((2, 4), (4,)),
         ((2, 3, 4), (1, 4)),
    ])
    def test_broadcastable_shape(self, shape_t, shape_c):
        tensor = torch.zeros(shape_t)
        module = EnsureShapeAndDtype(shape_c, broadcastable=True)
        module(tensor)

    @pytest.mark.parametrize('shape_t, shape_c', [
         ((3,), (2,)),
         ((2, 3), (2, 2)),
         ((2, 4), (3, 4)),
         ((2, 4), (3,)),
         ((2, 3, 4), (2, 2, 1)),
    ])
    def test_nonbroadcastable_shape(self, shape_t, shape_c):
        tensor = torch.zeros(shape_t)
        module = EnsureShapeAndDtype(shape_c, broadcastable=True)
        with pytest.raises(ValueError, match='non broadcastable'):
            module(tensor)

    @pytest.mark.parametrize('shape_t, shape_c', [
         ((2,), (None,)),
         ((2, 2), (2, None)),
         ((2, 1), (None, 2)),
         ((1, 4), (None, 4)),
         ((2, 3, 4), (2, None, 4)),
    ])
    def test_unknown_shape(self, shape_t, shape_c):
        tensor = torch.zeros(shape_t)
        module = EnsureShapeAndDtype(shape_c)
        module(tensor)

    @pytest.mark.parametrize('shape_t, shape_c', [
         ((3, 2), (2, None)),
         ((1, 4), (None, 2)),
         ((2, 3, 4), (3, None, 4)),
    ])
    def test_invalid_unknown_shape(self, shape_t, shape_c):
        tensor = torch.zeros(shape_t)
        module = EnsureShapeAndDtype(shape_c)
        with pytest.raises(ValueError, match='non broadcastable'):
            module(tensor)

    @pytest.mark.parametrize(
        'dtype', [torch.int32, torch.float32, torch.complex64]
    )
    def test_valid_dtypes(self, dtype):
        tensor = torch.zeros(1, dtype=dtype)
        module = EnsureShapeAndDtype(None, dtype)
        module(tensor)
        ensure_shape_and_dtype(tensor, None, dtype)

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
        ensure_shape_and_dtype(tensor, shape, dtype)

    # Too many warnings to list them all
    @pytest.mark.filterwarnings('ignore')
    def test_jit_module(self):
        shape = (10, 5)
        dtype = torch.float32
        tensor = torch.zeros(shape, dtype=dtype)
        module = EnsureShapeAndDtype(shape, dtype)
        jit_module = torch.jit.trace(module, (tensor,))
        jit_module(tensor)

        # An invalid tensor also passes, because checks are disabled
        shape = (5, 5)
        tensor = torch.zeros(shape, dtype=dtype)
        jit_module(tensor)

        # Tracing with a different shape fails during trace process
        with pytest.raises(ValueError, match='input shape is'):
            jit_module = torch.jit.trace(module, (tensor,))

    def test_torchscript_module(self):
        shape = (10, 5)
        dtype = torch.float32
        tensor = torch.zeros(shape, dtype=dtype)
        module = EnsureShapeAndDtype(shape, dtype)
        jit_module = torch.jit.script(module)
        jit_module(tensor)

        shape = (5, 5)
        tensor = torch.zeros(shape, dtype=dtype)
        # torchscript changes the exception type
        with pytest.raises(torch.jit.Error, match='input shape is'):
            jit_module(tensor)
