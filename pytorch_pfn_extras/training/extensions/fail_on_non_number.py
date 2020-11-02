import torch

from pytorch_pfn_extras.training import extension


class FailOnNonNumber(extension.Extension):
    """Trainer extension to raise RuntimeError if parameters contain NaN or Inf.
    Although parameters including non-number such as NaN and Inf are
    unnecessary in most cases the training loop will continue
    to compute even if the parameters in a given optimizer diverge.
    This extension is aimed to reduce unnecessary computations by throwing
    ``RuntimeError`` if the parameters contain NaN or Inf.
    """

    def __call__(self, manager):
        for name, model in manager.models.items():
            for param in model.parameters():
                if (not torch.isfinite(param).all()
                        or (param.grad is not None
                            and not torch.isfinite(param.grad).all())):
                    raise RuntimeError(
                        'Kill the process since parameters in optimizer'
                        ' \'{}\' diverge. R.I.P.'.format(name))
