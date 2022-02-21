import tempfile

import pytest

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions


params = [
    ([None, 4.0, 4.5, 3.0, 3.5],
     [4.0, 4.0, 3.0, 3.0],
     [1, 1, 3, 3],
     [3, 3, 9, 9],
     extensions.MinValue),
    ([None, 3.0, 4.5, 4.0, 5.0],
     [3.0, 4.5, 4.5, 5.0],
     [1, 2, 2, 4],
     [3, 6, 6, 12],
     extensions.MaxValue),
]


@pytest.mark.parametrize('observed_values,expected_best_values,'
                         'expected_best_epochs,expected_best_iterations,BestValueT',
                         params)
def test_best_observation(observed_values, expected_best_values,
                          expected_best_epochs, expected_best_iterations, BestValueT):
    max_epochs = 4
    iters_per_epoch = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ppe.training.ExtensionsManager(
            {}, {}, max_epochs, iters_per_epoch=iters_per_epoch, out_dir=tmpdir)

        def observer_fn(manager):
            return observed_values[manager.epoch]

        observe = extensions.observe_value('value', observer_fn)
        manager.extend(observe, trigger=(1, 'epoch'))

        best_value = BestValueT('value')
        manager.extend(best_value, trigger=(1, 'epoch'))

        for epoch in range(max_epochs):
            for _ in range(iters_per_epoch):
                with manager.run_iteration():
                    pass
            assert best_value.best_value == expected_best_values[epoch]
            assert best_value.best_epoch == expected_best_epochs[epoch]
            assert best_value.best_iteration == expected_best_iterations[epoch]

        # Save/Load state dict (snapshot support)
        assert best_value.state_dict() == {
            '_best_trigger': {
                '_best_value': expected_best_values[-1],
                '_summary': {},
                'interval_trigger': {}
            },
            '_best_it': expected_best_iterations[-1],
            '_best_epoch': expected_best_epochs[-1],
        }

        best_value2 = BestValueT('value')
        best_value2.load_state_dict(best_value.state_dict())
        assert best_value2.best_value == expected_best_values[-1]
        assert best_value2.best_iteration == expected_best_iterations[-1]
        assert best_value2.best_epoch == expected_best_epochs[-1]


def test_key_error():
    max_epochs = 4
    iters_per_epoch = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ppe.training.ExtensionsManager(
            {}, {}, max_epochs, iters_per_epoch=iters_per_epoch, out_dir=tmpdir)

        best_observation = extensions.BestValue('value', lambda a, b: a < b)
        manager.extend(best_observation, trigger=(1, 'epoch'))

        with pytest.raises(KeyError) as e:
            for _ in range(max_epochs):
                for _ in range(iters_per_epoch):
                    with manager.run_iteration():
                        pass
        assert 'Key "value" not found in the observation' in str(e.value)


def test_error_before_first_call():
    best_observation = extensions.BestValue('value', lambda a, b: a < b)
    with pytest.raises(RuntimeError):
        best_observation.best_value
    with pytest.raises(RuntimeError):
        best_observation.best_epoch
    with pytest.raises(RuntimeError):
        best_observation.best_iteration
