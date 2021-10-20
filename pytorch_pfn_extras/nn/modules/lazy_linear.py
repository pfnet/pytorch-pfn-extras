from typing import Any, Optional

import torch

from pytorch_pfn_extras.nn.modules.lazy import UninitializedParameter
from pytorch_pfn_extras.nn.modules.lazy import LazyInitializationMixin


class LazyLinear(LazyInitializationMixin, torch.nn.Linear):  # type: ignore[misc]
    """Linear module with lazy weight initialization.

    When ``in_features`` is ``None``, it is determined at the first time of
    the forward step.
    """

    lazy_parameter_names = ('weight',)

    def __init__(self, in_features: Optional[int], *args: Any, **kwargs: Any) -> None:
        super().__init__(in_features or 0, *args, **kwargs)
        if in_features is None:
            self.in_features = None  # type: ignore[assignment]
            self.weight = UninitializedParameter()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if isinstance(self.weight, UninitializedParameter):
            self.in_features = input.shape[-1]
            self.weight = torch.nn.Parameter(self.weight.new_empty(
                self.out_features, self.in_features))
            self.reset_parameters()
        return super().forward(input)

    def reset_parameters(self) -> None:
        # Defer initialization of parameters until shape of the parameter
        # is determiend.
        if self.lazy_parmeters_determined:
            super().reset_parameters()
