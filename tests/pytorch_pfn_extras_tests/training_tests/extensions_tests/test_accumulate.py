import json
import os
import random
import tempfile
from math import isclose, isnan
from statistics import mean, pstdev, stdev
from typing import Any, Callable, Dict, List

import pytest
import pytorch_pfn_extras as ppe
import torch
import torch.distributed
from pytorch_pfn_extras import distributed
from pytorch_pfn_extras.training.extensions.accumulate._accumulate_base import (
    AccumulateBase,
)
from pytorch_pfn_extras.training.trigger import Trigger, get_trigger

ITERATION_LENGTH = 2 * 3 * 5
NUM_EPOCH = 5
NUM_ITERATION_PER_EPOCH = ITERATION_LENGTH // NUM_EPOCH
epoch_trigger_case = {
    "value": [random.uniform(0, 100) for _ in range(ITERATION_LENGTH)],
    "trigger": (1, "epoch"),
}
float_epoch_trigger_case = {
    "value": [random.uniform(0, 100) for _ in range(ITERATION_LENGTH)],
    "trigger": (0.25, "epoch"),
}
epoch_iteration_trigger_case = {
    "value": [random.uniform(0, 100) for _ in range(ITERATION_LENGTH)],
    "trigger": (NUM_ITERATION_PER_EPOCH, "iteration"),
}
primary_iteration_trigger_case = {
    "value": [random.uniform(0, 100) for _ in range(ITERATION_LENGTH)],
    "trigger": (7, "iteration"),
}


def _init_distributed(use_cuda):
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        size, rank, local_rank = distributed.initialize_ompi_environment(
            backend="nccl", init_method="env"
        )
    else:
        pytest.skip("This test requires MPI to run")

    device = torch.device("cuda:{}".format(local_rank) if use_cuda else "cpu")
    torch.cuda.set_device(device)

    return size, rank, local_rank, device


def check_accumulate_extension(
    extension: AccumulateBase,
    expected_fn: Callable[..., float],
    value_list: List[float],
    trigger: Trigger,
    allow_nan: bool = False,
    distributed: bool = False,
) -> None:
    manager = ppe.training.ExtensionsManager(
        {}, [], NUM_EPOCH, iters_per_epoch=NUM_ITERATION_PER_EPOCH
    )
    manager.extend(extension=extension, trigger=(1, "iteration"))
    i = 0
    epoch_value_list = []
    for _ in range(NUM_EPOCH):
        for _ in range(NUM_ITERATION_PER_EPOCH):
            current_value = value_list[i]
            i += 1
            epoch_value_list.append(current_value)
            with manager.run_iteration():
                ppe.reporting.report({"value": current_value})
            if trigger(manager=manager):
                accumulated_value = float(
                    manager.observation["value/accumulated"]
                )
                try:
                    if distributed:
                        world_size = torch.distributed.get_world_size()
                        epoch_value_list_list = [None] * world_size
                        torch.distributed.all_gather_object(
                            epoch_value_list_list, epoch_value_list
                        )
                        epoch_value_list = sum(epoch_value_list_list, [])
                    expected_value = expected_fn(epoch_value_list)
                except Exception:
                    if allow_nan:
                        expected_value = float("nan")
                    else:
                        raise RuntimeError
                epoch_value_list.clear()
                assert (
                    isnan(accumulated_value) and isnan(expected_value)
                ) or isclose(
                    accumulated_value,
                    expected_value,
                    rel_tol=1e-9,
                    abs_tol=1e-6,
                )


@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_average_accumulate(case: Dict[str, Any]):
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.AverageAccumulate(
        conversion_key_pair=("value", "value/accumulated"), trigger=trigger
    )
    check_accumulate_extension(
        extension=extension,
        expected_fn=mean,
        value_list=case["value"],
        trigger=trigger,
    )


@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_standard_deviation_accumulate(case: Dict[str, Any]):
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.StandardDeviationAccumulate(
        conversion_key_pair=("value", "value/accumulated"), trigger=trigger
    )
    check_accumulate_extension(
        extension=extension,
        expected_fn=pstdev,
        value_list=case["value"],
        trigger=trigger,
    )


@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_unbiased_standard_deviation_accumulate(case: Dict[str, Any]):
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.UnbiasedStandardDeviationAccumulate(
        conversion_key_pair=("value", "value/accumulated"), trigger=trigger
    )
    check_accumulate_extension(
        extension=extension,
        expected_fn=stdev,
        value_list=case["value"],
        trigger=trigger,
        allow_nan=True,
    )


@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_min_accumulate(case: Dict[str, Any]):
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.MinAccumulate(
        conversion_key_pair=("value", "value/accumulated"), trigger=trigger
    )
    check_accumulate_extension(
        extension=extension,
        expected_fn=min,
        value_list=case["value"],
        trigger=trigger,
    )


@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_max_accumulate(case: Dict[str, Any]):
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.MaxAccumulate(
        conversion_key_pair=("value", "value/accumulated"), trigger=trigger
    )
    check_accumulate_extension(
        extension=extension,
        expected_fn=max,
        value_list=case["value"],
        trigger=trigger,
    )


@pytest.mark.skipif(
    not ppe.requires("1.11.0"),
    reason="Because torch<1.11 does not allow the use of TORCH_DISTRIBUTED_DEBUG=DETAIL and nccl communicator together.",
)
@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_average_accumulate_distributed(case: Dict[str, Any]):
    _init_distributed(use_cuda=True)
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.AverageAccumulate(
        conversion_key_pair=("value", "value/accumulated"),
        trigger=trigger,
        distributed=True,
    )
    check_accumulate_extension(
        extension=extension,
        expected_fn=mean,
        value_list=case["value"],
        trigger=trigger,
        distributed=True,
    )


@pytest.mark.skipif(
    not ppe.requires("1.11.0"),
    reason="Because torch<1.11 does not allow the use of TORCH_DISTRIBUTED_DEBUG=DETAIL and nccl communicator together.",
)
@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_standard_deviation_accumulate_distributed(case: Dict[str, Any]):
    _init_distributed(use_cuda=True)
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.StandardDeviationAccumulate(
        conversion_key_pair=("value", "value/accumulated"),
        trigger=trigger,
        distributed=True,
    )
    check_accumulate_extension(
        extension=extension,
        expected_fn=pstdev,
        value_list=case["value"],
        trigger=trigger,
        distributed=True,
    )


@pytest.mark.skipif(
    not ppe.requires("1.11.0"),
    reason="Because torch<1.11 does not allow the use of TORCH_DISTRIBUTED_DEBUG=DETAIL and nccl communicator together.",
)
@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_unbiased_standard_deviation_accumulate_distributed(
    case: Dict[str, Any]
):
    _init_distributed(use_cuda=True)
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.UnbiasedStandardDeviationAccumulate(
        conversion_key_pair=("value", "value/accumulated"),
        trigger=trigger,
        distributed=True,
    )
    check_accumulate_extension(
        extension=extension,
        expected_fn=stdev,
        value_list=case["value"],
        trigger=trigger,
        allow_nan=True,
        distributed=True,
    )


@pytest.mark.skipif(
    not ppe.requires("1.11.0"),
    reason="Because torch<1.11 does not allow the use of TORCH_DISTRIBUTED_DEBUG=DETAIL and nccl communicator together.",
)
@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_min_accumulate_distributed(case: Dict[str, Any]):
    _init_distributed(use_cuda=True)
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.MinAccumulate(
        conversion_key_pair=("value", "value/accumulated"),
        trigger=trigger,
        distributed=True,
    )
    check_accumulate_extension(
        extension=extension,
        expected_fn=min,
        value_list=case["value"],
        trigger=trigger,
        distributed=True,
    )


@pytest.mark.skipif(
    not ppe.requires("1.11.0"),
    reason="Because torch<1.11 does not allow the use of TORCH_DISTRIBUTED_DEBUG=DETAIL and nccl communicator together.",
)
@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_max_accumulate_distributed(case: Dict[str, Any]):
    _init_distributed(use_cuda=True)
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.MaxAccumulate(
        conversion_key_pair=("value", "value/accumulated"),
        trigger=trigger,
        distributed=True,
    )
    check_accumulate_extension(
        extension=extension,
        expected_fn=max,
        value_list=case["value"],
        trigger=trigger,
        distributed=True,
    )


def check_accumulate_extension_with_log_report(
    extension: AccumulateBase,
    expected_fn: Callable[..., float],
    value_list: List[float],
    trigger: Trigger,
    allow_nan: bool = False,
    distributed: bool = False,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = ppe.training.ExtensionsManager(
            {},
            [],
            NUM_EPOCH,
            iters_per_epoch=NUM_ITERATION_PER_EPOCH,
            out_dir=tmp_dir,
        )
        manager.extend(extension=extension, trigger=(1, "iteration"))
        log_report = ppe.training.extensions.LogReport(
            filename="out", format="json", trigger=trigger
        )
        manager.extend(extension=log_report, trigger=(1, "iteration"))

        i = 0
        trigger_count = 0
        epoch_value_list = []
        for _ in range(NUM_EPOCH):
            for _ in range(NUM_ITERATION_PER_EPOCH):
                current_value = value_list[i]
                i += 1
                epoch_value_list.append(current_value)
                with manager.run_iteration():
                    ppe.reporting.report({"value": current_value})
                if trigger(manager=manager):
                    trigger_count += 1
                    with open(os.path.join(tmp_dir, "out")) as f:
                        log_values = json.load(f)
                    assert isinstance(log_values, list)
                    assert len(log_values) == trigger_count
                    last_value = log_values[-1]
                    assert isinstance(last_value, dict)
                    assert "value/accumulated" in last_value
                    accumulated_value = last_value["value/accumulated"]
                    try:
                        if distributed:
                            world_size = torch.distributed.get_world_size()
                            epoch_value_list_list = [None] * world_size
                            torch.distributed.all_gather_object(
                                epoch_value_list_list, epoch_value_list
                            )
                            epoch_value_list = sum(epoch_value_list_list, [])
                        expected_value = expected_fn(epoch_value_list)
                    except Exception:
                        if allow_nan:
                            expected_value = float("nan")
                        else:
                            raise RuntimeError
                    epoch_value_list.clear()
                    assert (
                        allow_nan
                        and isnan(accumulated_value)
                        and isnan(expected_value)
                    ) or isclose(
                        accumulated_value,
                        expected_value,
                        rel_tol=1e-9,
                        abs_tol=1e-6,
                    )


@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_average_accumulate_with_log_report(case: Dict[str, Any]):
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.AverageAccumulate(
        conversion_key_pair=("value", "value/accumulated"), trigger=trigger
    )
    check_accumulate_extension_with_log_report(
        extension=extension,
        expected_fn=mean,
        value_list=case["value"],
        trigger=trigger,
    )


@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_standard_deviation_accumulate_with_log_report(case: Dict[str, Any]):
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.StandardDeviationAccumulate(
        conversion_key_pair=("value", "value/accumulated"), trigger=trigger
    )
    check_accumulate_extension_with_log_report(
        extension=extension,
        expected_fn=pstdev,
        value_list=case["value"],
        trigger=trigger,
    )


@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_unbiased_standard_deviation_accumulate_with_log_report(
    case: Dict[str, Any]
):
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.UnbiasedStandardDeviationAccumulate(
        conversion_key_pair=("value", "value/accumulated"), trigger=trigger
    )
    check_accumulate_extension_with_log_report(
        extension=extension,
        expected_fn=stdev,
        value_list=case["value"],
        trigger=trigger,
        allow_nan=True,
    )


@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_min_accumulate_with_log_report(case: Dict[str, Any]):
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.MinAccumulate(
        conversion_key_pair=("value", "value/accumulated"), trigger=trigger
    )
    check_accumulate_extension_with_log_report(
        extension=extension,
        expected_fn=min,
        value_list=case["value"],
        trigger=trigger,
    )


@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_max_accumulate_with_log_report(case: Dict[str, Any]):
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.MaxAccumulate(
        conversion_key_pair=("value", "value/accumulated"), trigger=trigger
    )
    check_accumulate_extension_with_log_report(
        extension=extension,
        expected_fn=max,
        value_list=case["value"],
        trigger=trigger,
    )


@pytest.mark.skipif(
    not ppe.requires("1.11.0"),
    reason="Because torch<1.11 does not allow the use of TORCH_DISTRIBUTED_DEBUG=DETAIL and nccl communicator together.",
)
@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_average_accumulate_with_log_report_distributed(case: Dict[str, Any]):
    _init_distributed(use_cuda=True)
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.AverageAccumulate(
        conversion_key_pair=("value", "value/accumulated"),
        trigger=trigger,
        distributed=True,
    )
    check_accumulate_extension_with_log_report(
        extension=extension,
        expected_fn=mean,
        value_list=case["value"],
        trigger=trigger,
        distributed=True,
    )


@pytest.mark.skipif(
    not ppe.requires("1.11.0"),
    reason="Because torch<1.11 does not allow the use of TORCH_DISTRIBUTED_DEBUG=DETAIL and nccl communicator together.",
)
@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_standard_deviation_accumulate_with_log_report_distributed(
    case: Dict[str, Any]
):
    _init_distributed(use_cuda=True)
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.StandardDeviationAccumulate(
        conversion_key_pair=("value", "value/accumulated"),
        trigger=trigger,
        distributed=True,
    )
    check_accumulate_extension_with_log_report(
        extension=extension,
        expected_fn=pstdev,
        value_list=case["value"],
        trigger=trigger,
        distributed=True,
    )


@pytest.mark.skipif(
    not ppe.requires("1.11.0"),
    reason="Because torch<1.11 does not allow the use of TORCH_DISTRIBUTED_DEBUG=DETAIL and nccl communicator together.",
)
@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_unbiased_standard_deviation_accumulate_with_log_report_distributed(
    case: Dict[str, Any]
):
    _init_distributed(use_cuda=True)
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.UnbiasedStandardDeviationAccumulate(
        conversion_key_pair=("value", "value/accumulated"),
        trigger=trigger,
        distributed=True,
    )
    check_accumulate_extension_with_log_report(
        extension=extension,
        expected_fn=stdev,
        value_list=case["value"],
        trigger=trigger,
        allow_nan=True,
        distributed=True,
    )


@pytest.mark.skipif(
    not ppe.requires("1.11.0"),
    reason="Because torch<1.11 does not allow the use of TORCH_DISTRIBUTED_DEBUG=DETAIL and nccl communicator together.",
)
@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_min_accumulate_with_log_report_distributed(case: Dict[str, Any]):
    _init_distributed(use_cuda=True)
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.MinAccumulate(
        conversion_key_pair=("value", "value/accumulated"),
        trigger=trigger,
        distributed=True,
    )
    check_accumulate_extension_with_log_report(
        extension=extension,
        expected_fn=min,
        value_list=case["value"],
        trigger=trigger,
        distributed=True,
    )


@pytest.mark.skipif(
    not ppe.requires("1.11.0"),
    reason="Because torch<1.11 does not allow the use of TORCH_DISTRIBUTED_DEBUG=DETAIL and nccl communicator together.",
)
@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.parametrize(
    "case",
    [
        epoch_trigger_case,
        float_epoch_trigger_case,
        epoch_iteration_trigger_case,
        primary_iteration_trigger_case,
    ],
)
def test_max_accumulate_with_log_report_distributed(case: Dict[str, Any]):
    _init_distributed(use_cuda=True)
    trigger = get_trigger(case["trigger"])
    extension = ppe.training.extensions.MaxAccumulate(
        conversion_key_pair=("value", "value/accumulated"),
        trigger=trigger,
        distributed=True,
    )
    check_accumulate_extension_with_log_report(
        extension=extension,
        expected_fn=max,
        value_list=case["value"],
        trigger=trigger,
        distributed=True,
    )
