import torch.utils.checkpoint


class _CheckpointFunction(torch.utils.checkpoint.CheckpointFunction):
    """Checkpoint a model or part of the model with BN support.

    Refer to https://pytorch.org/docs/stable/checkpoint.html
    for detailed information.
    When using checkpointing in model using BatchNormalization, the
    momentum is updated twice, while we only need one update to ensure
    correctness.
    Using `ppe.utils.checkpointing.checkpoint` as a drop-in replacement
    can help deal with incorrect values in the BatchNormalization
    persistent parameters.
    """
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        _patch_bn_momentum(run_function)
        return super(_CheckpointFunction, _CheckpointFunction).forward(
            ctx, run_function, preserve_rng_state, *args)


def _patch_bn_momentum(module):
    if not hasattr(module, '_bn_momentum_patched'):
        if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
            return
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            # Set momentum so that two forward passes will produce the same
            # EMA as one forward pass.
            module.momentum = 1 - (1 - module.momentum) ** 0.5
        for _, child in module.named_children():
            _patch_bn_momentum(child)
    module._bn_momentum_patched = True


def checkpoint(function, *args, **kwargs):
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError(
            'Unexpected keyword arguments: ' + ','.join(arg for arg in kwargs))
    return _CheckpointFunction.apply(function, preserve, *args)
