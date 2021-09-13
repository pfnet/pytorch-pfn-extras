import pytest
import torch

from pytorch_pfn_extras.nn import EnsureShape


class TestEnsureShape:
    @pytest.mark.parametrize(
        'shape', [(), (1,), (1, 1), (2,), (2, 4), (2, 3, 4)]
    )
    def test_valid_shape(self, shape):
        tensor = torch.zeros(shape)
        module = EnsureShape(shape)
        module(tensor)

    @pytest.mark.parametrize(
        'shape', [(), (1,), (1, 1), (2,), (2, 4), (2, 3, 4)]
    )
    def test_invalid_shape(self, shape):
        tensor = torch.zeros((1, 2, 3))
        module = EnsureShape(shape)
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
        module = EnsureShape(shape_c, broadcastable=True)
        module(tensor)

    @pytest.mark.parametrize('shape_t, shape_c', [
         ((3,), (2,)),
         ((2, 3), (2, 2)),
         ((2, 4), (3, 4)),
         ((2, 3, 4), (2, 2, 1)),
    ])
    def test_nonbroadcastable_shape(self, shape_t, shape_c):
        tensor = torch.zeros(shape_t)
        module = EnsureShape(shape_c, broadcastable=True)
        with pytest.raises(ValueError, match='non broadcastable'):
            module(tensor)
