import torch

from pytorch_pfn_extras.nn.modules.lazy import LazyInitializationMixin
from pytorch_pfn_extras.nn.modules.lazy import UninitializedParameter


class _LazyConvNd(LazyInitializationMixin):

    lazy_parameter_names = ('weight',)

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels or 0, *args, **kwargs)
        if in_channels is None:
            self.in_channels = None
            self.weight = UninitializedParameter()

    def forward(self, input):
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
        return super().forward(input)

    def reset_parameters(self):
        # Defer initialization of parameters until shape of all parameters
        # are ready.
        if self.lazy_parmeters_determined:
            super().reset_parameters()


class LazyConv1d(_LazyConvNd, torch.nn.Conv1d):
    """Conv1d module with lazy weight initialization.

    When ``in_channels`` is ``None``, it is determined at the first time of
    the forward step.
    """
    pass


class LazyConv2d(_LazyConvNd, torch.nn.Conv2d):
    """Conv2d module with lazy weight initialization.

    When ``in_channels`` is ``None``, it is determined at the first time of
    the forward step.
    """
    pass


class LazyConv3d(_LazyConvNd, torch.nn.Conv3d):
    """Conv3d module with lazy weight initialization.

    When ``in_channels`` is ``None``, it is determined at the first time of
    the forward step.
    """
    pass
