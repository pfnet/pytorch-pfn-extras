from typing import Any, Tuple, Optional

import torch


class EnsureShape(torch.nn.Module):
    """Module to check the shape of a tensor.

    Args:
       shape: Tuple with the desired shape. If the input tensor shape
          is not compatible, `ValueError` will be raised.
       broadcastable: Check if the shape is different, check
          if the shapes are compatible
    """

    def __init__(
            self,
            shape: Tuple[int],
            *args: Any,
            broadcastable: Optional[bool] = False,
            **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self._shape = shape
        self._broadcastable = broadcastable

    def forward(self, input: torch.Tensor):
        if input.shape != self._shape:
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
        return input
