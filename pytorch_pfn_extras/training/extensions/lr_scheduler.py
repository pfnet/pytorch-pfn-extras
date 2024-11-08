from typing import Any, Dict, Optional

from pytorch_pfn_extras._torch_version import requires
from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training import trigger as trigger_module
from pytorch_pfn_extras.training._manager_protocol import (
    ExtensionsManagerProtocol,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau


def _get_value_from_log_report(
    manager: ExtensionsManagerProtocol, key: Any
) -> Any:
    # Find and return the latest reported "key" from LogReport
    if key is None:
        return None
    if key not in manager.observation:
        raise ValueError(
            "{} is not found in the reported values {}".format(
                key, manager.observation
            )
        )

    return manager.observation[key]


def _default_stepper(
    manager: ExtensionsManagerProtocol, scheduler: Any
) -> None:
    if isinstance(scheduler, ReduceLROnPlateau):
        LRScheduler.step_by_value("val/loss")(manager, scheduler)
    else:
        scheduler.step()


def check_optimizer_is_called(optimizer: Optimizer) -> bool:
    if requires("2.4.0.dev"):
        # https://github.com/pytorch/pytorch/blob/afda6685ae87cce7ac2fe4bac3926572da2960f7/torch/optim/lr_scheduler.py#L172-L191
        # TODO: Rewrite this URL when pytorch 2.4.0 is released.
        if hasattr(optimizer.step, "_wrapped_by_lr_sched"):
            return getattr(optimizer, "_opt_called", False)
        else:
            return True
    else:
        # https://github.com/pytorch/pytorch/blob/v2.0.1/torch/optim/lr_scheduler.py#L137-L138
        if hasattr(optimizer.step, "_with_counter"):
            return bool(optimizer._step_count >= 1)  # type: ignore[attr-defined]
        else:
            return True


class LRScheduler(extension.Extension):
    """Trainer extension to adjust the learning rate using PyTorch's learning
    rate scheduler.

    This extension calls `step()` method of the given LR scheduler.
    (`torch.option.lr_scheduler.*`). When using `ReduceLROnPlateau`, the
    latest reported `val/loss` value will be used. This behavior can be
    customized by passing a custom `stepper` function.

    Args:
        scheduler (_LRScheduler or ReduceLROnPlateau): Any instance of
            `torch.optim.lr_scheduler.*`.
        stepper (callable): Function that performs the step on
            the scheduler.
        trigger: Frequency to call this extension.
        wait_for_first_optimizer_step (bool): Wait until `optimizer.step()` is called
            before invoking `scheduler.step()`. This can address the issue where
            `optimizer.step()` is not called from the first iteration when using GradScaler.
    """

    def __init__(
        self,
        scheduler: Any,
        *,
        stepper: Any = _default_stepper,
        trigger: trigger_module.TriggerLike = (1, "epoch"),
        wait_for_first_optimizer_step: bool = False,
        is_async: bool = True,
    ) -> None:
        self.scheduler = scheduler
        self.trigger = trigger_module.get_trigger(trigger)
        self.stepper = stepper
        self.wait_for_first_optimizer_step = wait_for_first_optimizer_step
        self.is_async = is_async

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        # https://github.com/pytorch/pytorch/blob/v2.0.1/torch/optim/lr_scheduler.py#L137-L138
        if (
            self.wait_for_first_optimizer_step
            and not check_optimizer_is_called(self.scheduler.optimizer)
        ):
            return
        self.stepper(manager, self.scheduler)

    @staticmethod
    def step_by_value(key: Optional[str]) -> Any:
        def _stepper(
            manager: ExtensionsManagerProtocol, scheduler: Any
        ) -> None:
            scheduler.step(_get_value_from_log_report(manager, key))

        return _stepper

    def state_dict(self) -> Dict[str, Any]:
        return {"scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state["scheduler"])
