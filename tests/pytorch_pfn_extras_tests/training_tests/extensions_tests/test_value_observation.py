import torch

import pytorch_pfn_extras as ppe


def test_observe_value():
    lr = 0.1
    manager = ppe.training.ExtensionsManager({}, [], 1, iters_per_epoch=1)
    extension = ppe.training.extensions.observe_value("lr", lambda x: lr)
    manager.extend(extension)
    with manager.run_iteration():
        pass

    assert manager.observation["lr"] == lr


def test_observe_lr():
    lr = 0.01
    manager = ppe.training.ExtensionsManager({}, [], 1, iters_per_epoch=1)
    optimizer = torch.optim.Adam({torch.nn.Parameter()}, lr=lr)
    extension = ppe.training.extensions.observe_lr(optimizer)
    manager.extend(extension)
    with manager.run_iteration():
        pass

    assert manager.observation["lr"] == lr
