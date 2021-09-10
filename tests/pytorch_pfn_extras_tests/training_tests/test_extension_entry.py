import torch

import pytorch_pfn_extras as ppe


def _get_dummy_manager():
    model = torch.nn.Module()
    return ppe.training.ExtensionsManager(
        {'main': model},
        [],  # optimizers
        10,  # max_epochs
        iters_per_epoch=1,
    )


def test_default_name():
    class MyExtension(ppe.training.Extension):
        name = None
        default_name = 'defalut_name'

    ext = MyExtension()
    entry = ppe.training.ExtensionEntry(ext)
    assert entry.name == MyExtension.default_name
    entry = ppe.training.ExtensionEntry(ext, name='updated')
    assert entry.name == 'updated'


def test_name():
    class MyExtension(ppe.training.Extension):
        name = 'name'
        default_name = 'defalut_name'

    ext = MyExtension()
    entry = ppe.training.ExtensionEntry(ext)
    assert entry.name == MyExtension.name
    entry = ppe.training.ExtensionEntry(ext, name='updated')
    assert entry.name == 'updated'


def test_priority():
    class MyExtension(ppe.training.Extension):
        priority = 100

    ext = MyExtension()
    entry = ppe.training.ExtensionEntry(ext)
    assert entry.priority == MyExtension.priority
    entry = ppe.training.ExtensionEntry(ext, priority=10)
    assert entry.priority == 10


def test_trigger():
    class MyExtension(ppe.training.Extension):
        trigger = (1, 'iteration')

    ext = MyExtension()
    entry = ppe.training.ExtensionEntry(ext)
    assert isinstance(entry.trigger, ppe.training.triggers.IntervalTrigger)
    assert entry.trigger.period == 1
    assert entry.trigger.unit == 'iteration'
    entry = ppe.training.ExtensionEntry(ext, trigger=(3, 'epoch'))
    assert isinstance(entry.trigger, ppe.training.triggers.IntervalTrigger)
    assert entry.trigger.period == 3
    assert entry.trigger.unit == 'epoch'
