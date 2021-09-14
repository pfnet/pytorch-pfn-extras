from typing import Optional, Tuple

import torch


class EnsureShapeAndDtype(torch.nn.Module):
    """Module to check the shape of a tensor.

    Args:
       shape: Tuple with the desired shape. If the input tensor shape
           is not compatible, `ValueError` will be raised. If `None` is set
           as a dimension value, that dimension will be ignored.
       dtype: Checks if the `dtype` of the input thensor matches the
           provided one.
       broadcastable: Check if the shapes are compatible using broadcasting
           rules.
       can_cast: Check if the input tensor can be casted to the provided type.
    """

    def __init__(
            self,
            shape: Optional[Tuple[int]] = None,
            dtype: Optional[torch.dtype] = None,
            *,
            broadcastable: Optional[bool] = False,
            can_cast: Optional[bool] = False,
    ):
        super().__init__()  # type: ignore[no-untyped-call]
        if shape is None and dtype is None:
            raise ValueError(
                'shape, dtype or both arguments must be specified')
        self._dtype = dtype
        self._broadcastable = broadcastable
        self._can_cast = can_cast
        # Check if there are Nones in the shape and replace them by 1s
        # so we can compare the shapes using broadcast semantics
        c_shape = None
        if shape is not None and None in shape:
            c_shape = tuple(
                x if x is not None else 1  # type: ignore[unreachable]
                for x in shape
            )
            self._broadcastable = True
        else:
            c_shape = shape

        self._shape = None
        if c_shape is not None:
            # This is required for torch script
            self._shape = torch.tensor(c_shape)

    def _broadcast(self, shape_1: torch.Tensor, shape_2: torch.Tensor) -> bool:
        # Torch broadcast_shapes raises an exception, we want a simple
        # method that returs True/False so we can use torch script
        l_1, l_2 = len(shape_1), len(shape_2)
        if l_1 != l_2:
            # we would like to do checks like the below one
            # but torchscript converts everything to tensors
            # shape_2 = (1,) * (l_1 - l_2) + shape_2
            if l_1 < l_2:
                shape_1 = torch.cat([torch.ones(l_2 - l_1), shape_1])
            else:
                shape_2 = torch.cat([torch.ones(l_1 - l_2), shape_2])
        # Look for Elements that are different in the tensors
        for a, b in zip(shape_1, shape_2):
            if a != b and a != 1 and b != 1:
                return False
        return True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # To make it compatible with torchscript since torch.Size does not work
        t_shape = torch.tensor(input.shape)
        if self._shape is not None and list(t_shape) != list(self._shape):
            if self._broadcastable:
                if not self._broadcast(t_shape, self._shape):
                    raise ValueError(
                        f'Shapes {self._shape} and {input.shape} are non'
                        ' broadcastable')
            else:
                raise ValueError(
                    f'Expected {self._shape}, input shape is {input.shape}')
        if self._dtype is not None and input.dtype != self._dtype:
            if self._can_cast:
                if not torch.can_cast(input.dtype, self._dtype):
                    raise ValueError(
                        f'Input dtype {input.dtype} can\'t be casted to'
                        f' {self._dtype}')
            else:
                raise ValueError(
                    f'Expected {self._dtype}, input dtype is {input.dtype}')
        return input


def ensure_shape_and_dtype(
        tensor: torch.Tensor,
        shape: Optional[Tuple[int]],
        dtype: Optional[torch.dtype] = None,
        broadcastable: Optional[bool] = False,
        can_cast: Optional[bool] = False
) -> None:
    """Checks the shape and type of a tensor.

    Args:
       shape: Tuple with the desired shape. If the input tensor shape
           is not compatible, `ValueError` will be raised. If `None` is set
           as a dimension value, that dimension will be ignored.
       dtype: Checks if the `dtype` of the input thensor matches the
           provided one.
       broadcastable: Check if the shapes are compatible using broadcasting
           rules.
       can_cast: Check if the input tensor can be casted to the provided type.
    """
    EnsureShapeAndDtype(
        shape, dtype,
        broadcastable=broadcastable, can_cast=can_cast)(tensor)
