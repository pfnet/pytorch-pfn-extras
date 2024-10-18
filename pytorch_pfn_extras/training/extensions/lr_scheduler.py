from typing import Any, Dict, Optional

import pytorch_pfn_extras
import pytorch_pfn_extras._torch_version
from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training import trigger as trigger_module
from pytorch_pfn_extras.training._manager_protocol import (
    ExtensionsManagerProtocol,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

if pytorch_pfn_extras._torch_version.requires("2.0.0"):
    from torch.optim.lr_scheduler import LRScheduler as _LRScheduler  # type: ignore[attr-defined]  # isort:skip
else:
    from torch.optim.lr_scheduler import _LRScheduler


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
        self.has_opt_called = pytorch_pfn_extras.requires(
            "2.4.0"
        ) and isinstance(self.scheduler, _LRScheduler)
        self.trigger = trigger_module.get_trigger(trigger)
        self.stepper = stepper
        self.wait_for_first_optimizer_step = wait_for_first_optimizer_step
        self.is_async = is_async

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        # https://github.com/pytorch/pytorch/blob/v2.0.1/torch/optim/lr_scheduler.py#L137-L138
        # https://github.com/pytorch/pytorch/blob/v2.4.1/torch/optim/lr_scheduler.py#L215
        if self.wait_for_first_optimizer_step and (
            (
                hasattr(self.scheduler.optimizer.step, "_with_counter")
                and self.scheduler.optimizer._step_count < 1
            )
            or (
                self.has_opt_called
                and not getattr(self.scheduler.optimizer, "_opt_called", False)
            )
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
