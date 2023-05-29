import pytest
from pytorch_pfn_extras import training
from pytorch_pfn_extras.training import triggers


def _test_trigger(manager, trigger, key, accuracies, expected):
    for accuracy, e in zip(accuracies, expected):
        with manager.run_iteration():
            manager.observation = {key: accuracy}
        assert trigger(manager) == e


def _compare(best_value, new_value):
    return abs(new_value) < abs(best_value)


_trigger_test_params = [
    # interval = 1 iterations
    (
        triggers.MaxValueTrigger,
        ((1, "iteration"),),
        1,
        [0.5, 0.5, 0.4, 0.6],
        [True, False, False, True],
        1,
    ),
    (
        triggers.MinValueTrigger,
        ((1, "iteration"),),
        1,
        [0.5, 0.5, 0.4, 0.6],
        [True, False, True, False],
        1,
    ),
    # interval = 2 iterations
    (
        triggers.MaxValueTrigger,
        ((2, "iteration"),),
        1,
        [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        [False, True, False, False, False, False, False, True],
        2,
    ),
    (
        triggers.MinValueTrigger,
        ((2, "iteration"),),
        1,
        [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        [False, True, False, False, False, True, False, False],
        2,
    ),
    # interval = 2 iterations, unaligned resume
    (
        triggers.MaxValueTrigger,
        ((2, "iteration"),),
        1,
        [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        [False, True, False, False, False, False, False, True],
        3,
    ),
    (
        triggers.MinValueTrigger,
        ((2, "iteration"),),
        1,
        [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        [False, True, False, False, False, True, False, False],
        3,
    ),
    # interval = 1 epoch, 1 epoch = 2 iterations
    (
        triggers.MaxValueTrigger,
        ((1, "epoch"),),
        2,
        [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        [False, True, False, False, False, False, False, True],
        2,
    ),
    (
        triggers.MinValueTrigger,
        ((1, "epoch"),),
        2,
        [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        [False, True, False, False, False, True, False, False],
        2,
    ),
    # interval = 1 epoch, 1 epoch = 2 iterations, unaligned resume
    (
        triggers.MaxValueTrigger,
        ((1, "epoch"),),
        2,
        [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        [False, True, False, False, False, False, False, True],
        3,
    ),
    (
        triggers.MinValueTrigger,
        ((1, "epoch"),),
        2,
        [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        [False, True, False, False, False, True, False, False],
        3,
    ),
    # best_value trigger test
    (
        triggers.BestValueTrigger,
        (_compare, (1, "iteration")),
        2,
        [0.5, -0.5, -0.6, 0.6, 0.4, -0.4, -0.3, 0.3],
        [True, False, False, False, True, False, True, False],
        3,
    ),
]


@pytest.mark.parametrize(
    "trigger_type,trigger_args,iters_per_epoch,accuracies,expected,resume",
    _trigger_test_params,
)
def test_trigger(
    trigger_type, trigger_args, iters_per_epoch, accuracies, expected, resume
):
    key = "main/accuracy"
    manager = training.ExtensionsManager(
        {}, [], 100, iters_per_epoch=iters_per_epoch
    )
    trigger = trigger_type(key, *trigger_args)
    _test_trigger(manager, trigger, key, accuracies, expected)


@pytest.mark.parametrize(
    "trigger_type,trigger_args,iters_per_epoch,accuracies,expected,resume",
    _trigger_test_params,
)
def test_resumed_trigger(
    trigger_type, trigger_args, iters_per_epoch, accuracies, expected, resume
):
    key = "main/accuracy"
    manager = training.ExtensionsManager(
        {}, [], 100, iters_per_epoch=iters_per_epoch
    )

    trigger = trigger_type(key, *trigger_args)
    _test_trigger(manager, trigger, key, accuracies[:resume], expected[:resume])

    state = trigger.state_dict()

    new_trigger = trigger_type(key, *trigger_args)
    new_trigger.load_state_dict(state)
    _test_trigger(
        manager, new_trigger, key, accuracies[resume:], expected[resume:]
    )
