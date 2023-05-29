import random

import pytest
import pytorch_pfn_extras as ppe

_parametrize = pytest.mark.parametrize(
    "iters_per_epoch,call_on_resume,resume",
    [
        # basic
        (5, False, 4),
        # call on resume
        (5, True, 4),
    ],
)


@_parametrize
def test_trigger(iters_per_epoch, call_on_resume, resume):
    del resume  # resume is ignored
    expected = [True] + [False] * 6
    finished = [False] + [True] * 6
    manager = ppe.training.ExtensionsManager(
        {}, [], 100, iters_per_epoch=iters_per_epoch
    )
    trigger = ppe.training.triggers.OnceTrigger(call_on_resume)
    for e, f in zip(expected, finished):
        assert trigger.finished == f
        assert trigger(manager) == e
        with manager.run_iteration():
            pass


@_parametrize
def test_resumed_trigger(iters_per_epoch, call_on_resume, resume):
    expected = [True] + [False] * 6
    finished = [False] + [True] * 6
    if call_on_resume:
        expected[resume] = True
        finished[resume] = False
    manager = ppe.training.ExtensionsManager(
        {}, [], 100, iters_per_epoch=iters_per_epoch
    )
    trigger = ppe.training.triggers.OnceTrigger(call_on_resume)
    for e, f in zip(expected[:resume], finished[:resume]):
        with manager.run_iteration():
            pass
        assert trigger.finished == f
        assert trigger(manager) == e
    state = trigger.state_dict()

    trigger2 = ppe.training.triggers.OnceTrigger(call_on_resume)
    trigger2.load_state_dict(state)
    for e, f in zip(expected[resume:], finished[resume:]):
        with manager.run_iteration():
            pass
        assert trigger2.finished == f
        assert trigger2(manager) == e


@_parametrize
def test_trigger_sparse_call(iters_per_epoch, call_on_resume, resume):
    del resume  # resume is ignored
    expected = [True] + [False] * 6
    finished = [False] + [True] * 6
    for _ in range(10):
        manager = ppe.training.ExtensionsManager(
            {}, [], 100, iters_per_epoch=iters_per_epoch
        )
        trigger = ppe.training.triggers.OnceTrigger(call_on_resume)
        accumulated = False
        accumulated_finished = True
        for e, f in zip(expected, finished):
            with manager.run_iteration():
                accumulated = accumulated or e
                accumulated_finished = accumulated_finished and f
                if random.randrange(2):
                    assert trigger.finished == accumulated_finished
                    assert trigger(manager) == accumulated
                    accumulated = False
                    accumulated_finished = True
