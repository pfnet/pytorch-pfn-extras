import pathlib

import numpy
import pytorch_pfn_extras as ppe
import torch


def _test_trigger(trigger, key, accuracies, expected, tmp_path: pathlib.Path):
    manager = ppe.training.ExtensionsManager(
        {}, [], 100, iters_per_epoch=1, out_dir=str(tmp_path)
    )
    for a, e in zip(accuracies, expected):
        with manager.run_iteration():
            pass
        manager.observation = {key: a}
        assert trigger(manager) == e


def test_early_stopping_trigger_with_accuracy(tmp_path: pathlib.Path):
    key = "main/accuracy"
    trigger = ppe.training.triggers.EarlyStoppingTrigger(
        monitor=key, patience=3, check_trigger=(1, "epoch"), verbose=False
    )
    accuracies = [
        torch.Tensor(numpy.asarray(acc, dtype=numpy.float32))
        for acc in [0.5, 0.5, 0.6, 0.7, 0.6, 0.4, 0.3, 0.2]
    ]
    expected = [False, False, False, False, False, False, True, True]
    _test_trigger(trigger, key, accuracies, expected, tmp_path)


def test_early_stopping_trigger_with_loss(tmp_path: pathlib.Path):
    key = "main/loss"
    trigger = ppe.training.triggers.EarlyStoppingTrigger(
        monitor=key, patience=3, check_trigger=(1, "epoch")
    )
    accuracies = [
        torch.Tensor(numpy.asarray(acc, dtype=numpy.float32))
        for acc in [100, 80, 30, 10, 20, 24, 30, 35]
    ]
    expected = [False, False, False, False, False, False, True, True]
    _test_trigger(trigger, key, accuracies, expected, tmp_path)


def test_early_stopping_trigger_with_max_epoch(tmp_path: pathlib.Path):
    key = "main/loss"
    trigger = ppe.training.triggers.EarlyStoppingTrigger(
        monitor=key,
        patience=3,
        check_trigger=(1, "epoch"),
        max_trigger=(3, "epoch"),
    )
    accuracies = [
        torch.Tensor(numpy.asarray(acc, dtype=numpy.float32))
        for acc in [100, 80, 30]
    ]
    expected = [False, False, True]
    _test_trigger(trigger, key, accuracies, expected, tmp_path)


def test_early_stopping_trigger_with_max_iteration(tmp_path: pathlib.Path):
    key = "main/loss"
    trigger = ppe.training.triggers.EarlyStoppingTrigger(
        monitor=key,
        patience=3,
        check_trigger=(1, "epoch"),
        max_trigger=(3, "iteration"),
    )
    accuracies = [
        torch.Tensor(numpy.asarray(acc, dtype=numpy.float32))
        for acc in [100, 80, 30]
    ]

    expected = [False, False, True]
    _test_trigger(trigger, key, accuracies, expected, tmp_path)
