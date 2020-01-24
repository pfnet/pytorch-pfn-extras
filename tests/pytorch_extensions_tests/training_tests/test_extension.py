import pytest
import torch

import pytorch_extensions as pe


def _get_dummy_manager():
    model = torch.nn.Module()
    return pe.training.ExtensionsManager(
        {'main': model},
        [],  # optimizers
        10,  # max_epochs
        [],  # extensions
    )


def test_raise_error_if_call_not_implemented():
    class MyExtension(pe.training.Extension):
        pass

    ext = MyExtension()
    trainer = _get_dummy_manager()
    with pytest.raises(NotImplementedError):
        ext(trainer)


def test_default_name():
    class MyExtension(pe.training.Extension):
        pass

    ext = MyExtension()
    assert ext.default_name == 'MyExtension'


def test_deleted_invoke_before_training():
    class MyExtension(pe.training.Extension):
        pass

    ext = MyExtension()
    with pytest.raises(AttributeError):
        ext.invoke_before_training


def test_make_extension():
    def initialize(trainer):
        pass

    @pe.training.make_extension(trigger=(2, 'epoch'), default_name='my_ext',
                                priority=50, initializer=initialize)
    def my_extension(trainer):
        pass

    assert my_extension.trigger == (2, 'epoch')
    assert my_extension.default_name == 'my_ext'
    assert my_extension.priority == 50
    assert my_extension.initialize is initialize


def test_make_extension_default_values():
    @pe.training.make_extension()
    def my_extension(trainer):
        pass

    assert my_extension.trigger == (1, 'iteration')
    assert my_extension.default_name == 'my_extension'
    assert my_extension.priority == pe.training.PRIORITY_READER
    assert my_extension.initialize is None


def test_make_extension_unexpected_kwargs():
    with pytest.raises(TypeError):
        @pe.training.make_extension(foo=1)
        def my_extension(_):
            pass
