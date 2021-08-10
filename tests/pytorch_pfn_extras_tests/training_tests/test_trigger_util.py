import pytest

from pytorch_pfn_extras import training
from pytorch_pfn_extras.training import _trigger_util
from pytorch_pfn_extras.training import triggers


@pytest.mark.parametrize(
    'iters_per_epoch,trigger_args,expected',
    [
        # Never fire trigger
        (2, None, [False, False, False, False, False, False, False]),

        # Interval trigger
        (2, (2, 'iteration'),
         [False, True, False, True, False, True, False]),
        (2, (2, 'epoch'),
         [False, False, False, True, False, False, False]),

        # Callable object
        (2, _trigger_util.get_trigger(None),
         [False, False, False, False, False, False, False]),
        (2, triggers.IntervalTrigger(2, 'iteration'),
         [False, True, False, True, False, True, False]),
        (2, (lambda trainer: trainer.iteration == 3),
         [False, False, True, False, False, False, False]),
    ]
)
def test_get_trigger(iters_per_epoch, trigger_args, expected):
    trainer = training.ExtensionsManager(
        {}, [], 100, iters_per_epoch=iters_per_epoch)
    trigger = _trigger_util.get_trigger(trigger_args)

    # before the first iteration, trigger should be False
    for _, e in enumerate(expected):
        with trainer.run_iteration():
            pass
        assert trigger(trainer) == e
