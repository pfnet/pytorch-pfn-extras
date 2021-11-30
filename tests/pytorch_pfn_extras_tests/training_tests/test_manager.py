import pytest

import torch
from torch import nn

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras import training


def test_manager_status_info():
    manager = training.ExtensionsManager(
        nn.Module(),
        object(),
        10,
        iters_per_epoch=4
    )
    manager.iteration = 9
    assert manager.iteration == 9
    assert manager.epoch == 2
    assert manager.epoch_detail == 2.25

    manager.iteration = 15
    assert manager.epoch == 3
    assert manager.epoch_detail == 3.75


class _DummyExtension(training.Extension):

    def __init__(
            self, extension_id, call_record, init_record, use_model=False):
        self.extension_id = extension_id
        self.call_record = call_record
        self.init_record = init_record
        self.use_model = use_model

    def __call__(self, manager):
        self.call_record.append(self.extension_id)
        if self.use_model:
            # Check if models are accessible.
            assert manager.models['main'] is not None
            assert manager.raw_models['main'] is not None


class _DummyExtensionInitialize(_DummyExtension):

    def initialize(self, manager):
        self.init_record.append(self.extension_id)


def test_extensions_manager_extensions():
    model = nn.Module()
    optimizer = object()
    max_epochs = 5
    iters_per_epoch = 4
    manager = training.ExtensionsManager(
        {'model_name': model},
        {'optimizer_name': optimizer},
        max_epochs,
        iters_per_epoch=iters_per_epoch,
    )

    call_record = []
    init_record = []

    dummy5 = _DummyExtension(5, call_record, init_record)
    exts = [
        _DummyExtension(0, call_record, init_record),
        _DummyExtensionInitialize(1, call_record, init_record),
        _DummyExtension(2, call_record, init_record),
        _DummyExtensionInitialize(3, call_record, init_record),
        _DummyExtensionInitialize(4, call_record, init_record),
        lambda manager: dummy5(manager),
        training.ExtensionEntry(
            _DummyExtension(6, call_record, init_record),
            name='ext6', priority=-3, call_before_training=True,
        ),
        training.ExtensionEntry(
            _DummyExtension(7, call_record, init_record),
            name='ext7_', priority=-2, call_before_training=True,
        ),
    ]

    manager.extend(exts[0], 'ext0', priority=2, call_before_training=True)
    manager.extend(exts[1], 'ext1', priority=1, call_before_training=False)
    manager.extend(exts[2], 'ext2', priority=3, call_before_training=False)
    manager.extend(exts[3], 'ext3', priority=0, call_before_training=True)
    manager.extend(exts[4], 'ext4', priority=4, call_before_training=True)
    manager.extend(exts[5], 'ext5', priority=-1, call_before_training=True)
    manager.extend(exts[6])
    manager.extend(exts[7], 'ext7', priority=-4, call_before_training=False)

    assert manager.get_extension('ext0') is exts[0]
    assert manager.get_extension('ext1') is exts[1]
    assert manager.get_extension('ext2') is exts[2]
    assert manager.get_extension('ext3') is exts[3]
    assert manager.get_extension('ext4') is exts[4]
    assert manager.get_extension('ext6') is exts[6].extension
    assert manager.get_extension('ext7') is exts[7].extension

    with pytest.raises(ValueError):
        manager.get_extension('ext10')

    for it in range(max_epochs * iters_per_epoch):
        call_record.clear()
        init_record.clear()

        with manager.run_iteration():
            assert manager.iteration == it

            if it == 0:
                assert call_record == [4, 0, 3, 5, 6]
                assert init_record == [4, 1, 3]
            else:
                assert call_record == []
                assert init_record == []

            call_record.clear()
            init_record.clear()

        assert call_record == [4, 2, 0, 1, 3, 5, 6, 7]
        assert init_record == []


class _StateDictObj():
    def __init__(self, *, state_dict=None, state_dict_to_be_loaded=None):
        super().__init__()
        self.called_load_state_dict = 0
        self._state_dict = state_dict
        self._state_dict_to_be_loaded = state_dict_to_be_loaded

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, state_dict):
        self.called_load_state_dict += 1
        assert state_dict is self._state_dict_to_be_loaded


class _StateDictModel(_StateDictObj, nn.Module):

    def forward(self, *args):
        pass


class _StateDictOptimizer(_StateDictObj):

    def zero_grad(self):
        pass

    def step(self):
        pass


class _StateDictExtension(_StateDictObj, training.Extension):

    def __call__(self, manager):
        pass


def _fake_loss(*args):
    return torch.tensor([0.0], requires_grad=True)


def test_extensions_manager_state_dict():
    model_state_dict = object()
    optimizer_state_dict = object()
    extension_state_dict = object()
    max_epochs = 5
    iters_per_epoch = 4
    passed_iteration = 11

    manager = training.ExtensionsManager(
        {'model_name': _StateDictModel(state_dict=model_state_dict)},
        {'optimizer_name': _StateDictObj(state_dict=optimizer_state_dict)},
        max_epochs,
        iters_per_epoch=iters_per_epoch,
    )

    manager.extend(
        _StateDictExtension(
            state_dict=extension_state_dict), name='extension_name')

    for _ in range(passed_iteration):
        with manager.run_iteration():
            pass

    state_dict = manager.state_dict()

    assert state_dict == {
        'ppe_version': ppe.__version__,
        '_start_execution': passed_iteration,
        '_start_iteration': passed_iteration,
        'models': {'model_name': model_state_dict},
        'optimizers': {'optimizer_name': optimizer_state_dict},
        'extensions': {'extension_name': {
            'extension': extension_state_dict,
            'trigger': {
            },
        }},
    }

    new_model = _StateDictModel(state_dict_to_be_loaded=model_state_dict)
    new_optimizer = _StateDictObj(state_dict_to_be_loaded=optimizer_state_dict)
    new_extension = _StateDictExtension(
        state_dict_to_be_loaded=extension_state_dict)
    new_manager = training.ExtensionsManager(
        {'model_name': new_model},
        {'optimizer_name': new_optimizer},
        max_epochs,
        iters_per_epoch=iters_per_epoch,
    )
    new_manager.extend(new_extension, name='extension_name')
    new_manager.load_state_dict(state_dict)
    assert new_model.called_load_state_dict == 1
    assert new_optimizer.called_load_state_dict == 1
    assert new_optimizer.called_load_state_dict == 1


def test_extensions_manager_state_dict_old_ppe_no_version():
    model_state_dict = object()
    optimizer_state_dict = object()
    max_epochs = 5
    iters_per_epoch = 4
    passed_iteration = 11

    manager = training.ExtensionsManager(
        {'model_name': _StateDictModel(state_dict=model_state_dict)},
        {'optimizer_name': _StateDictObj(state_dict=optimizer_state_dict)},
        max_epochs,
        iters_per_epoch=iters_per_epoch,
    )

    for _ in range(passed_iteration):
        with manager.run_iteration():
            pass

    new_model = _StateDictModel(state_dict_to_be_loaded=model_state_dict)
    new_optimizer = _StateDictObj(state_dict_to_be_loaded=optimizer_state_dict)
    manager_2 = training.ExtensionsManager(
        {'model_name': new_model},
        {'optimizer_name': new_optimizer},
        max_epochs,
        iters_per_epoch=iters_per_epoch,
    )

    state_dict = manager.state_dict()
    del state_dict['ppe_version']
    with pytest.warns(UserWarning, match='version'):
        manager_2.load_state_dict(state_dict)


def test_extensions_manager_state_dict_old_ppe_version():
    model_state_dict = object()
    optimizer_state_dict = object()
    max_epochs = 5
    iters_per_epoch = 4
    passed_iteration = 11

    manager = training.ExtensionsManager(
        {'model_name': _StateDictModel(state_dict=model_state_dict)},
        {'optimizer_name': _StateDictObj(state_dict=optimizer_state_dict)},
        max_epochs,
        iters_per_epoch=iters_per_epoch,
    )

    for _ in range(passed_iteration):
        with manager.run_iteration():
            pass

    new_model = _StateDictModel(state_dict_to_be_loaded=model_state_dict)
    new_optimizer = _StateDictObj(state_dict_to_be_loaded=optimizer_state_dict)
    manager_2 = training.ExtensionsManager(
        {'model_name': new_model},
        {'optimizer_name': new_optimizer},
        max_epochs,
        iters_per_epoch=iters_per_epoch,
    )

    state_dict = manager.state_dict()
    state_dict['ppe_version'] = '0.4.0'
    with pytest.warns(UserWarning, match='version'):
        manager_2.load_state_dict(state_dict)


def test_extensions_manager_state_dict_future_ppe_version():
    model_state_dict = object()
    optimizer_state_dict = object()
    max_epochs = 5
    iters_per_epoch = 4
    passed_iteration = 11

    manager = training.ExtensionsManager(
        {'model_name': _StateDictModel(state_dict=model_state_dict)},
        {'optimizer_name': _StateDictObj(state_dict=optimizer_state_dict)},
        max_epochs,
        iters_per_epoch=iters_per_epoch,
    )

    for _ in range(passed_iteration):
        with manager.run_iteration():
            pass

    new_model = _StateDictModel(state_dict_to_be_loaded=model_state_dict)
    new_optimizer = _StateDictObj(state_dict_to_be_loaded=optimizer_state_dict)
    manager_2 = training.ExtensionsManager(
        {'model_name': new_model},
        {'optimizer_name': new_optimizer},
        max_epochs,
        iters_per_epoch=iters_per_epoch,
    )

    state_dict = manager.state_dict()
    state_dict['ppe_version'] = '23.0.0'
    with pytest.warns(UserWarning, match='version'):
        manager_2.load_state_dict(state_dict)


def test_ignite_extensions_manager_state_dict():

    try:
        from ignite.engine import create_supervised_trainer
    except ImportError:
        pytest.skip('pytorch-ignite not found')

    model_state_dict = object()
    optimizer_state_dict = object()
    extension_state_dict = object()
    max_epochs = 5
    iters_per_epoch = 4
    passed_iteration = 20

    model = _StateDictModel(state_dict=model_state_dict)
    optimizer = _StateDictOptimizer(state_dict=optimizer_state_dict)

    trainer = create_supervised_trainer(
        model, optimizer, _fake_loss)

    manager = training.IgniteExtensionsManager(
        trainer,
        {'model_name': model},
        {'optimizer_name': optimizer},
        max_epochs,
    )
    manager.extend(
        _StateDictExtension(
            state_dict=extension_state_dict), name='extension_name')

    loader = torch.utils.data.DataLoader(
        [(i, i) for i in range(iters_per_epoch)])
    trainer.run(loader, max_epochs=max_epochs)

    state_dict = manager.state_dict()

    assert state_dict == {
        'ppe_version': ppe.__version__,
        '_start_execution': passed_iteration,
        '_start_iteration': passed_iteration,
        '_epoch_length': iters_per_epoch,
        'models': {'model_name': model_state_dict},
        'optimizers': {'optimizer_name': optimizer_state_dict},
        'extensions': {'extension_name': {
            'extension': extension_state_dict,
            'trigger': {
            },
        }},
    }

    new_model = _StateDictModel(state_dict_to_be_loaded=model_state_dict)
    new_optimizer = _StateDictOptimizer(
        state_dict_to_be_loaded=optimizer_state_dict)
    new_extension = _StateDictExtension(
        state_dict_to_be_loaded=extension_state_dict)

    new_trainer = create_supervised_trainer(
        model, optimizer, _fake_loss)
    new_manager = training.IgniteExtensionsManager(
        new_trainer,
        {'model_name': new_model},
        {'optimizer_name': new_optimizer},
        max_epochs,
    )
    new_manager.extend(new_extension, name='extension_name')
    new_manager.load_state_dict(state_dict)
    assert new_model.called_load_state_dict == 1
    assert new_optimizer.called_load_state_dict == 1
    assert new_optimizer.called_load_state_dict == 1


def test_extensions_manager_with_plain_model_and_optimizer():
    model_state_dict = object()
    optimizer_state_dict = object()
    max_epochs = 5
    iters_per_epoch = 4
    manager = training.ExtensionsManager(
        _StateDictModel(state_dict=model_state_dict),
        _StateDictObj(state_dict=optimizer_state_dict),
        max_epochs,
        iters_per_epoch=iters_per_epoch,
    )

    state_dict = manager.state_dict()

    assert state_dict == {
        'ppe_version': ppe.__version__,
        '_start_execution': 0,
        '_start_iteration': 0,
        'models': {'main': model_state_dict},
        'optimizers': {'main': optimizer_state_dict},
        'extensions': {}
    }


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self._wrapper_module = model
        self.accessed = False

    def wrapper_module(self):
        self.accessed = True
        return self._wrapper_module


def test_model_transformations():
    model_state_dict = object()
    optimizer_state_dict = object()
    max_epochs = 5
    iters_per_epoch = 4
    model = Wrapper(_StateDictModel(state_dict=model_state_dict))
    manager = training.ExtensionsManager(
        model,
        _StateDictObj(state_dict=optimizer_state_dict),
        max_epochs,
        iters_per_epoch=iters_per_epoch,
        transform_model=lambda n, x: x.wrapper_module(),
    )

    assert not isinstance(manager.models['main'], Wrapper)
    assert model.accessed


def test_model_transformations_in_state_dict():
    model_state_dict = object()
    optimizer_state_dict = object()
    max_epochs = 5
    iters_per_epoch = 4
    model = Wrapper(_StateDictModel(state_dict=model_state_dict))
    manager = training.ExtensionsManager(
        model,
        _StateDictObj(state_dict=optimizer_state_dict),
        max_epochs,
        iters_per_epoch=iters_per_epoch,
        transform_model=lambda n, x: x.wrapper_module(),
    )

    state_dict = manager.state_dict()
    assert model.accessed

    new_model = _StateDictModel(state_dict_to_be_loaded=model_state_dict)
    new_optimizer = _StateDictObj(state_dict_to_be_loaded=optimizer_state_dict)
    new_manager = training.ExtensionsManager(
        Wrapper(new_model),
        new_optimizer,
        max_epochs,
        iters_per_epoch=iters_per_epoch,
        transform_model=lambda n, x: x.wrapper_module(),
    )
    new_manager.load_state_dict(state_dict)
    assert isinstance(new_manager.models['main'], _StateDictModel)


def test_call_optimizers():
    m = torch.nn.Linear(5, 5)
    a = torch.ones(1, requires_grad=True)
    optimizer = torch.optim.SGD(lr=1.0, params=[a])
    manager = training.ExtensionsManager(
        m,
        optimizer,
        1,
        iters_per_epoch=1,
    )
    with manager.run_iteration(step_optimizers=['main']):
        a.grad = torch.tensor([2.0])
    assert torch.equal(a.detach(), torch.tensor([-1.]))


def test_needs_state_this_iteration():
    m = torch.nn.Linear(5, 5)
    a = torch.ones(1, requires_grad=True)
    optimizer = torch.optim.SGD(lr=1.0, params=[a])
    extension = _DummyExtension(0, [], [], True)
    extension.name = 'Dummy'
    extension.needs_model_state = True
    extension.trigger = (50, 'iteration')
    manager = training.ExtensionsManager(
        m,
        optimizer,
        1,
        iters_per_epoch=100,
        extensions=[extension]
    )
    while not manager.stop_trigger:
        with manager.run_iteration():
            # iteration is always added 1 before calling
            # extensions
            if manager.iteration in (49, 99):
                assert manager.needs_state_this_iteration()
            else:
                assert not manager.needs_state_this_iteration()


@pytest.mark.parametrize('priority', [
    None,
    training.extension.PRIORITY_SNAPSHOT,
    training.PRIORITY_WRITER,
])
def test_extensions_accessing_models_without_flag(priority):
    m = torch.nn.Linear(5, 5)
    a = torch.ones(1, requires_grad=True)
    optimizer = torch.optim.SGD(lr=1.0, params=[a])
    extension = _DummyExtension(0, [], [], True)
    extension.name = 'Dummy'
    extension.needs_model_state = False
    extension.trigger = (1, 'iteration')
    if priority is not None:
        extension.priority = priority
    manager = training.ExtensionsManager(
        m,
        optimizer,
        1,
        iters_per_epoch=5,
        extensions=[extension]
    )
    while not manager.stop_trigger:
        with pytest.raises(RuntimeError):
            with manager.run_iteration():
                pass


def test_deferred_iteration():
    m = torch.nn.Linear(5, 5)
    a = torch.ones(1, requires_grad=True)
    optimizer = torch.optim.SGD(lr=1.0, params=[a])
    call_record = []
    extension = _DummyExtension(0, call_record, [])
    extension.name = 'Dummy 0'
    extension.trigger = (1, 'iteration')
    extension.is_async = True
    extension2 = _DummyExtension(1, call_record, [])
    extension2.name = 'Dummy 1'
    extension2.trigger = (1, 'iteration')
    manager = training.ExtensionsManager(
        m,
        optimizer,
        1,
        iters_per_epoch=100,
        extensions=[extension, extension2]
    )
    for _ in range(5):
        with manager.run_iteration() as iter_handler:
            # iteration is always added 1 before calling
            # extensions
            iter_handler.defer()
    with manager.run_iteration():
        pass

    assert manager.iteration == 1
    assert manager.execution == 6
    assert call_record == [0] * 6 + [1]

    for _ in range(5):
        with manager.complete_iteration():
            pass

    assert manager.iteration == 6
    assert manager.execution == 6
    assert call_record == [0] * 6 + [1] * 6


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
