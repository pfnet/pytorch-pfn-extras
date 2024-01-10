import pathlib
import tempfile
from unittest.mock import MagicMock

import pytest
import pytorch_pfn_extras as ppe
import torch


def _setup_manager(tmp_path: pathlib.Path):
    param = torch.nn.Parameter(torch.zeros(10))
    optim = torch.optim.SGD([param], 1.0)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=[1, 2, 3], gamma=0.1, last_epoch=-1
    )
    ext = ppe.training.extensions.LRScheduler(sched, trigger=(1, "iteration"))
    manager = ppe.training.ExtensionsManager(
        {},
        {"main": optim},
        1,
        extensions=[ext],
        iters_per_epoch=40,
        out_dir=str(tmp_path),
    )
    return optim, manager


def test_lr_scheduler(tmp_path: pathlib.Path):
    optim, manager = _setup_manager(tmp_path)
    for i in range(4):
        with manager.run_iteration(step_optimizers=["main"]):
            if i < 1:
                assert optim.param_groups[0]["lr"] == pytest.approx(1.0)
            elif i < 2:
                assert optim.param_groups[0]["lr"] == pytest.approx(1e-1)
            elif i < 3:
                assert optim.param_groups[0]["lr"] == pytest.approx(1e-2)
            elif i < 4:
                assert optim.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_serialize_scheduler(tmp_path: pathlib.Path):
    optim, manager = _setup_manager(tmp_path)
    for i in range(2):
        with manager.run_iteration(step_optimizers=["main"]):
            if i < 1:
                assert optim.param_groups[0]["lr"] == pytest.approx(1.0)
            else:
                assert optim.param_groups[0]["lr"] == pytest.approx(1e-1)

    state = manager.state_dict()

    optim, manager = _setup_manager(tmp_path)
    manager.load_state_dict(state)
    for i in range(2):
        with manager.run_iteration(step_optimizers=["main"]):
            if i < 1:
                assert optim.param_groups[0]["lr"] == pytest.approx(1e-2)
            else:
                assert optim.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_reduce_lr_on_plateau():
    param = torch.nn.Parameter(torch.zeros(10))
    optim = torch.optim.SGD([param], 1.0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1)
    ext = ppe.training.extensions.LRScheduler(sched, trigger=(1, "iteration"))

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ppe.training.ExtensionsManager(
            {},
            {"main": optim},
            1,
            extensions=[ext],
            iters_per_epoch=4,
            out_dir=tmpdir,
        )
        manager.extend(
            ppe.training.extensions.LogReport(
                filename=None, trigger=(1, "iteration")
            )
        )
        for _ in range(4):
            with manager.run_iteration():
                ppe.reporting.report({"val/loss": 1.0})
        lr = optim.param_groups[0]["lr"]
        assert lr == pytest.approx(1e-1)


def test_reduce_lr_on_plateau_no_report(tmp_path: pathlib.Path):
    param = torch.nn.Parameter(torch.zeros(10))
    optim = torch.optim.SGD([param], 1.0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1)
    ext = ppe.training.extensions.LRScheduler(sched, trigger=(1, "iteration"))

    manager = ppe.training.ExtensionsManager(
        {},
        {"main": optim},
        1,
        extensions=[ext],
        iters_per_epoch=4,
        out_dir=str(tmp_path),
    )
    with pytest.raises(ValueError):
        with manager.run_iteration():
            pass


def test_lr_scheduler_wait_for_first_optimizer_step(tmp_path: pathlib.Path):
    param = torch.nn.Parameter(torch.zeros(10))
    optim = torch.optim.SGD([param], 1.0)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=[1, 2, 3], gamma=0.1, last_epoch=-1
    )
    stepper = MagicMock()
    ext = ppe.training.extensions.LRScheduler(
        sched,
        stepper=stepper,
        wait_for_first_optimizer_step=True,
        trigger=(1, "iteration"),
    )
    manager = ppe.training.ExtensionsManager(
        {},
        {"main": optim},
        1,
        extensions=[ext],
        iters_per_epoch=40,
        out_dir=str(tmp_path),
    )
    for i in range(4):
        with manager.run_iteration():
            pass
    assert stepper.call_count == 0
    for i in range(4):
        with manager.run_iteration(step_optimizers=["main"]):
            pass
    assert stepper.call_count == 4
    for i in range(4):
        with manager.run_iteration():
            pass

    assert stepper.call_count == 8


def test_wait_for_first_optimizer_step_with_non_torch_lr_scheduler(
    tmp_path: pathlib.Path,
):
    param = torch.nn.Parameter(torch.zeros(10))
    optim = torch.optim.SGD([param], 1.0)
    sched = MagicMock()
    sched.optimizer = optim
    stepper = MagicMock()
    ext = ppe.training.extensions.LRScheduler(
        sched,
        stepper=stepper,
        wait_for_first_optimizer_step=True,
        trigger=(1, "iteration"),
    )
    manager = ppe.training.ExtensionsManager(
        {},
        {},
        1,
        extensions=[ext],
        iters_per_epoch=40,
        out_dir=str(tmp_path),
    )
    for i in range(4):
        with manager.run_iteration():
            pass
    assert stepper.call_count == 4
