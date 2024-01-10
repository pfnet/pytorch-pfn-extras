import pathlib

import pytorch_pfn_extras as ppe
import torch


def test_observe_value(tmp_path: pathlib.Path):
    lr = 0.1
    manager = ppe.training.ExtensionsManager(
        {}, [], 1, iters_per_epoch=1, out_dir=str(tmp_path)
    )
    extension = ppe.training.extensions.observe_value("lr", lambda x: lr)
    manager.extend(extension)
    with manager.run_iteration():
        pass

    assert manager.observation["lr"] == lr


def test_observe_lr(tmp_path: pathlib.Path):
    lr = 0.01
    manager = ppe.training.ExtensionsManager(
        {}, [], 1, iters_per_epoch=1, out_dir=str(tmp_path)
    )
    optimizer = torch.optim.Adam({torch.nn.Parameter()}, lr=lr)
    extension = ppe.training.extensions.observe_lr(optimizer)
    manager.extend(extension)
    with manager.run_iteration():
        pass

    assert manager.observation["lr"] == lr
