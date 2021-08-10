import pytest

from pytorch_pfn_extras import training
from pytorch_pfn_extras.training import triggers


_scheduled_trigger_test_params = [
    # single iteration
    (2, (2, 'iteration'),
     [False, True, False, False, False, False, False], 3),
    # multiple iteration
    (2, ([2, 4], 'iteration'),
     [False, True, False, True, False, False, False], 3),
    # single epoch
    (3, (1, 'epoch'), [False, False, True, False, False, False, False], 3),
    # multiple epoch
    (3, ([1, 2], 'epoch'), [False, False, True, False, False, True, False], 4),
    # single fractional epoch
    (2, (1.5, 'epoch'), [False, False, True, False, False, False, False], 4),
    # multiple fractional epoch
    (2, ([1.5, 2.5], 'epoch'),
     [False, False, True, False, True, False, False], 4),
    # TODO(imanishi): Restore these tests after supported.
    # # single unaligned epoch
    # (2.5, (1, 'epoch'), [False, False, True, False, False, False, False], 4),
    # # multiple unaligned epoch
    # (2.5, ([1, 2], 'epoch'),
    #  [False, False, True, False, True, False, False], 4),
    # # single tiny epoch
    # (0.5, (1, 'epoch'), [True, False, False, False, False, False, False], 4),
    # # multiple tiny epoch
    # (0.5, ([1, 2], 'epoch'),
    #  [True, False, False, False, False, False, False], 4),
]


def _test_trigger(trainer, trigger, expected):
    for e in expected:
        with trainer.run_iteration():
            pass
        assert trigger(trainer) == e


@pytest.mark.parametrize(
    'iters_per_epoch,schedule,expected,resume',
    _scheduled_trigger_test_params
)
def test_trigger(iters_per_epoch, schedule, expected, resume):
    trainer = training.ExtensionsManager(
        {}, [], 100, iters_per_epoch=iters_per_epoch)
    trigger = triggers.ManualScheduleTrigger(*schedule)

    _test_trigger(trainer, trigger, expected)


@pytest.mark.parametrize(
    'iters_per_epoch,schedule,expected,resume',
    _scheduled_trigger_test_params
)
def test_resumed_trigger(
        iters_per_epoch, schedule, expected, resume):
    trainer = training.ExtensionsManager(
        {}, [], 100, iters_per_epoch=iters_per_epoch)
    trigger = triggers.ManualScheduleTrigger(*schedule)

    _test_trigger(
        trainer, trigger,
        expected[:resume])

    state = trigger.state_dict()
    new_trigger = triggers.ManualScheduleTrigger(*schedule)
    new_trigger.load_state_dict(state)

    _test_trigger(trainer, new_trigger, expected[resume:])


def test_invalid_unit():
    with pytest.raises(ValueError):
        triggers.ManualScheduleTrigger(1, 'day')
