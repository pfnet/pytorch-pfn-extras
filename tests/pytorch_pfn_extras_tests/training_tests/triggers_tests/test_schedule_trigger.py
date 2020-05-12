import pytest

from pytorch_pfn_extras import training
from pytorch_pfn_extras.training import triggers


def expected_finished(pos, num):
    return [i >= pos for i in range(num)]


_scheduled_trigger_test_params = [
    # single iteration
    (2, (2, 'iteration'),
     [False, True, False, False, False, False, False],
     expected_finished(1, 7), 3),
    # multiple iteration
    (2, ([2, 4], 'iteration'),
     [False, True, False, True, False, False, False],
     expected_finished(3, 7), 3),
    # single epoch
    (3, (1, 'epoch'), [False, False, True, False, False, False, False],
     expected_finished(2, 7), 3),
    # multiple epoch
    (3, ([1, 2], 'epoch'), [False, False, True, False, False, True, False],
     expected_finished(5, 7), 4),
    # single fractional epoch
    (2, (1.5, 'epoch'), [False, False, True, False, False, False, False],
     expected_finished(2, 7), 4),
    # multiple fractional epoch
    (2, ([1.5, 2.5], 'epoch'),
     [False, False, True, False, True, False, False],
     expected_finished(4, 7), 4),
    # TODO(imanishi): Restore these tests after supported.
    # # single unaligned epoch
    # (2.5, (1, 'epoch'), [False, False, True, False, False, False, False],
    #  expected_finished(2, 7), 4),
    # # multiple unaligned epoch
    # (2.5, ([1, 2], 'epoch'),
    #  [False, False, True, False, True, False, False],
    #  expected_finished(4, 7), 4),
    # # single tiny epoch
    # (0.5, (1, 'epoch'), [True, False, False, False, False, False, False],
    #  expected_finished(0, 7), 4),
    # # multiple tiny epoch
    # (0.5, ([1, 2], 'epoch'),
    #  [True, False, False, False, False, False, False],
    #  expected_finished(0, 7), 4),
]


def _test_trigger(trainer, trigger, expected, finished):
    for (e, f) in zip(expected, finished):
        with trainer.run_iteration():
            pass
        assert trigger(trainer) == e
        assert trigger.finished == f


@pytest.mark.parametrize(
    'iters_per_epoch,schedule,expected,finished,resume',
    _scheduled_trigger_test_params
)
def test_trigger(iters_per_epoch, schedule, expected, finished, resume):
    trainer = training.ExtensionsManager(
        {}, [], 100, iters_per_epoch=iters_per_epoch)
    trigger = triggers.ManualScheduleTrigger(*schedule)

    _test_trigger(trainer, trigger, expected, finished)


@pytest.mark.parametrize(
    'iters_per_epoch,schedule,expected,finished,resume',
    _scheduled_trigger_test_params
)
def test_resumed_trigger(
        iters_per_epoch, schedule, expected, finished, resume):
    trainer = training.ExtensionsManager(
        {}, [], 100, iters_per_epoch=iters_per_epoch)
    trigger = triggers.ManualScheduleTrigger(*schedule)

    _test_trigger(
        trainer, trigger,
        expected[:resume], finished[:resume])

    state = trigger.state_dict()
    new_trigger = triggers.ManualScheduleTrigger(*schedule)
    new_trigger.load_state_dict(state)

    _test_trigger(trainer, new_trigger, expected[resume:], finished[resume:])


def test_invalid_unit():
    with pytest.raises(ValueError):
        triggers.ManualScheduleTrigger(1, 'day')
