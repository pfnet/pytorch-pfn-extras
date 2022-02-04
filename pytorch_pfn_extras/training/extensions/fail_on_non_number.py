import torch

from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol


class FailOnNonNumber(extension.Extension):
    """An extension to raise RuntimeError if parameters and its gradients
    contain NaN or Inf.

    Although parameters including non-number such as NaN and Inf are
    unnecessary in most cases the training loop will continue
    to compute even if the parameters in a given optimizer diverge.
    This extension is aimed to reduce unnecessary computations by throwing
    ``RuntimeError`` if the parameters contain NaN or Inf.

    Args:
        check_grad: Set to False to skip checking gradients.
    """

    needs_model_state = True

    def __init__(self, *, check_grad: bool = True):
        self._check_grad = check_grad

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        for name, model in manager.models.items():
            for param in model.parameters():
                if (not torch.isfinite(param).all()
                        or (self._check_grad
                            and param.grad is not None
                            and not torch.isfinite(param.grad).all())):
                    raise RuntimeError(
                        'Kill the process since parameters in optimizer'
                        ' \'{}\' diverge. R.I.P.'.format(name))
