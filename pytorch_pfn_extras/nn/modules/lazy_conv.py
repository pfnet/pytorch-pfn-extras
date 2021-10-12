from typing import Any, Optional

import torch

from pytorch_pfn_extras.nn.modules.lazy import (
    LazyInitializationMixin, UninitializedParameter
)


class _LazyConvNd(LazyInitializationMixin):

    lazy_parameter_names = ('weight',)

    def __init__(
            self: Any, in_channels: Optional[int], *args: Any, **kwargs: Any) -> None:
        super().__init__(in_channels or 0, *args, **kwargs)
        if in_channels is None:
            self.in_channels: Optional[int] = None
            self.weight = UninitializedParameter()

    def forward(self: Any, input: torch.Tensor) -> torch.Tensor:
        if isinstance(self.weight, UninitializedParameter):
            self.in_channels = input.shape[1]
            if self.transposed:
                shape = (self.in_channels, self.out_channels // self.groups,
                         *self.kernel_size)
            else:
                shape = (self.out_channels, self.in_channels // self.groups,
                         *self.kernel_size)
            self.weight = torch.nn.Parameter(self.weight.new_empty(*shape))
            self.reset_parameters()
        return super().forward(input)  # type: ignore

    def reset_parameters(self: Any) -> None:
        # Defer initialization of parameters until shape of all parameters
        # are ready.
        if self.lazy_parmeters_determined:
            super().reset_parameters()  # type: ignore[misc]


class LazyConv1d(_LazyConvNd, torch.nn.Conv1d):  # type: ignore[misc]
    """Conv1d module with lazy weight initialization.

    When ``in_channels`` is ``None``, it is determined at the first time of
    the forward step.
    """
    pass


class LazyConv2d(_LazyConvNd, torch.nn.Conv2d):  # type: ignore[misc]
    """Conv2d module with lazy weight initialization.

    When ``in_channels`` is ``None``, it is determined at the first time of
    the forward step.
    """
    pass


class LazyConv3d(_LazyConvNd, torch.nn.Conv3d):  # type: ignore[misc]
    """Conv3d module with lazy weight initialization.

    When ``in_channels`` is ``None``, it is determined at the first time of
    the forward step.
    """
    pass
