import pytest
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


def test_raise_error_if_call_not_implemented():
    class MyExtension(ppe.training.Extension):
        pass

    ext = MyExtension()
    trainer = _get_dummy_manager()
    with pytest.raises(NotImplementedError):
        ext(trainer)


def test_default_name():
    class MyExtension(ppe.training.Extension):
        pass

    ext = MyExtension()
    assert ext.default_name == 'MyExtension'


def test_deleted_invoke_before_training():
    class MyExtension(ppe.training.Extension):
        pass

    ext = MyExtension()
    with pytest.raises(AttributeError):
        ext.invoke_before_training


def test_make_extension():
    def initialize(trainer):
        pass

    @ppe.training.make_extension(trigger=(2, 'epoch'), default_name='my_ext',
                                 priority=50, initializer=initialize)
    def my_extension(trainer):
        pass

    assert my_extension.trigger == (2, 'epoch')
    assert my_extension.default_name == 'my_ext'
    assert my_extension.priority == 50
    assert my_extension.initialize is initialize


def test_make_extension_default_values():
    @ppe.training.make_extension()
    def my_extension(trainer):
        pass

    assert my_extension.trigger == (1, 'iteration')
    assert my_extension.default_name == 'my_extension'
    assert my_extension.priority == ppe.training.PRIORITY_READER
    assert my_extension.initialize is None


def test_make_extension_unexpected_kwargs():
    with pytest.raises(TypeError):
        @ppe.training.make_extension(foo=1)
        def my_extension(_):
            pass
