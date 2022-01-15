import tempfile

import pytest

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions


params = [
    # Minimize
    ([None, 4.0, 4.5, 3.0, 3.5],
     [4.0, 4.0, 3.0, 3.0],
     [1, 1, 3, 3],
     [3, 3, 9, 9],
     'MINIMIZE'),
    # Maximize
    ([None, 3.0, 4.5, 4.0, 5.0],
     [3.0, 4.5, 4.5, 5.0],
     [1, 2, 2, 4],
     [3, 6, 6, 12],
     'MAXIMIZE'),
]


@pytest.mark.parametrize('observed_values,expected_best_values,'
                         'expected_best_epochs,expected_best_iterations,direction',
                         params)
def test_best_observation(observed_values, expected_best_values,
                          expected_best_epochs, expected_best_iterations, direction):
    max_epochs = 4
    iters_per_epoch = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ppe.training.ExtensionsManager(
            {}, {}, max_epochs, iters_per_epoch=iters_per_epoch, out_dir=tmpdir)

        def observer_fn(manager):
            return observed_values[manager.epoch]

        observe = extensions.observe_value('value', observer_fn)
        manager.extend(observe, trigger=(1, 'epoch'))

        best_observation = extensions.BestObservation('value', direction)
        manager.extend(best_observation, trigger=(1, 'epoch'))

        for epoch in range(max_epochs):
            for _ in range(iters_per_epoch):
                with manager.run_iteration():
                    pass
            assert best_observation.best_value == expected_best_values[epoch]
            assert best_observation.best_epoch == expected_best_epochs[epoch]
            assert best_observation.best_iteration == expected_best_iterations[epoch]

        # Save/Load state dict (snapshot support)
        assert best_observation.state_dict() == {
            '_direction': direction,
            '_best_value': expected_best_values[-1],
            '_best_iteration': expected_best_iterations[-1],
            '_best_epoch': expected_best_epochs[-1],
        }

        best_observation2 = extensions.BestObservation('value', direction)
        best_observation2.load_state_dict(best_observation.state_dict())
        assert best_observation2._direction == direction
        assert best_observation2._best_value == expected_best_values[-1]
        assert best_observation2._best_it == expected_best_iterations[-1]
        assert best_observation2._best_epoch == expected_best_epochs[-1]


def test_key_error():
    max_epochs = 4
    iters_per_epoch = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ppe.training.ExtensionsManager(
            {}, {}, max_epochs, iters_per_epoch=iters_per_epoch, out_dir=tmpdir)

        best_observation = extensions.BestObservation('value', 'MINIMIZE')
        manager.extend(best_observation, trigger=(1, 'epoch'))

        with pytest.raises(RuntimeError) as e:
            for _ in range(max_epochs):
                for _ in range(iters_per_epoch):
                    with manager.run_iteration():
                        pass
        assert 'Key "value" not found in the observation' in str(e.value)


def test_error_before_first_call():
    best_observation = extensions.BestObservation('value', 'MINIMIZE')
    with pytest.raises(RuntimeError):
        best_observation.best_value
    with pytest.raises(RuntimeError):
        best_observation.best_epoch
    with pytest.raises(RuntimeError):
        best_observation.best_iteration


def test_direction_format():
    # Case insensitive
    extensions.BestObservation('value', 'minimize')
    extensions.BestObservation('value', 'Minimize')
    extensions.BestObservation('value', 'MiNiMiZe')
    extensions.BestObservation('value', 'maximize')
    extensions.BestObservation('value', 'Maximize')
    extensions.BestObservation('value', 'MaXiMiZe')

    with pytest.raises(ValueError):
        extensions.BestObservation('value', 'MMINIMIZEE')
    with pytest.raises(ValueError):
        extensions.BestObservation('value', 'MMAXIMIZEE')


def test_direction_default():
    best_observation = extensions.BestObservation('value')
    assert best_observation._direction == 'MINIMIZE'
