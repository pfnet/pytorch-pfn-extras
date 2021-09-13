from typing import Any, Optional, Tuple

import torch


class EnsureShapeAndDtype(torch.nn.Module):
    """Module to check the shape of a tensor.

    Args:
       shape: Tuple with the desired shape. If the input tensor shape
          is not compatible, `ValueError` will be raised.
       dtype: Checks if the `dtype` of the input thensor matches the
          provided one.
       broadcastable: Check if the shape is different, check
          if the shapes are compatible
       can_cast: Check if the input tensor can be casted to the provided type
    """

    def __init__(
            self,
            shape: Optional[Tuple[int]] = None,
            dtype: Optional[torch.dtype] = None,
            *args: Any,
            broadcastable: Optional[bool] = False,
            can_cast: Optional[bool] = False,
            **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        if shape is dtype is None:
            raise ValueError(
                'shape, dtype or both arguments must be specified')
        self._shape = shape
        self._dtype = dtype
        self._broadcastable = broadcastable
        self._can_cast = can_cast

    def forward(self, input: torch.Tensor):
        if self._shape is not None and input.shape != self._shape:
            if self._broadcastable:
                try:
                    torch.broadcast_shapes(self._shape, input.shape)
                except RuntimeError:
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
