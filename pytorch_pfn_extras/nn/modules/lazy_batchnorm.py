import torch

from pytorch_pfn_extras.nn.modules.lazy import LazyInitializationMixin
from pytorch_pfn_extras.nn.modules.lazy import UninitializedParameter


class _LazyBatchNorm(
    LazyInitializationMixin,
    torch.nn.modules.batchnorm._BatchNorm
):

    lazy_parameter_names = ('weight', 'bias')

    def __init__(self, num_features, *args, **kwargs):
        super().__init__(num_features or 0, *args, **kwargs)
        if not self.affine:
            raise ValueError(
                'LazyBatchNorm is not compatible with affine=False.'
                ' Use the regular BatchNorm layers instead')
        # weight and bias are registered in the mixin
        if num_features is None:
            self.num_features = None
            if self.track_running_stats:
                # these buffers are not always needed
                # so we avoid explicit initializations
                self.lazy_buffer_names = ('running_mean', 'running_var')

    def reset_parameters(self) -> None:
        if self.lazy_parmeters_determined:
            super().reset_parameters()

    def forward(self, input):
        if isinstance(self.weight, UninitializedParameter):
            self.num_features = input.shape[-1]
            if self.affine:
                self.weight = torch.nn.Parameter(
                    self.weight.new_empty(self.num_features)
                )
                self.bias = torch.nn.Parameter(
                    self.weight.new_empty(self.num_features)
                )
            if self.track_running_stats:
                self.running_mean = torch.zeros(
                    self.num_features, device=self.running_mean.device,
                    dtype=self.running_mean.dtype
                )
                self.running_var = torch.ones(
                    self.num_features, device=self.running_var.device,
                    dtype=self.running_mean.dtype
                )
            self.reset_parameters()
        return super().forward(input)


class LazyBatchNorm1d(_LazyBatchNorm, torch.nn.BatchNorm1d):
    """BatchNorm1d module with lazy weight initialization.

    When ``num_features`` is ``None``, it is determined at the first time of
    the forward step.
    """
    pass


class LazyBatchNorm2d(_LazyBatchNorm, torch.nn.BatchNorm2d):
    """BatchNorm2d module with lazy weight initialization.

    When ``num_features`` is ``None``, it is determined at the first time of
    the forward step.
    """
    pass


class LazyBatchNorm3d(_LazyBatchNorm, torch.nn.BatchNorm3d):
    """BatchNorm3d module with lazy weight initialization.

    When ``num_features`` is ``None``, it is determined at the first time of
    the forward step.
    """
    pass
