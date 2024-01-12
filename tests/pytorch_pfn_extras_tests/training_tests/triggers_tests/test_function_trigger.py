import pathlib
from unittest.mock import MagicMock

import pytest
from pytorch_pfn_extras.training import ExtensionsManager
from pytorch_pfn_extras.training import trigger as trigger_module
from pytorch_pfn_extras.training._trigger_util import TriggerLike
from pytorch_pfn_extras.training.triggers import FunctionTrigger


def test_function_is_called(tmp_path: pathlib.Path) -> None:
    fn = MagicMock()
    args = [MagicMock()]
    kwargs = {"a": MagicMock()}
    trigger = FunctionTrigger(
        fn=fn, args=args, kwargs=kwargs, trigger=(1, "iteration")
    )
    fn.assert_not_called()
    manager = ExtensionsManager(
        {}, {}, 1, iters_per_epoch=10, out_dir=str(tmp_path)
    )
    with manager.run_iteration():
        pass
    trigger(manager)
    fn.assert_called_once_with(*args, **kwargs)


def test_trigger_with_value(tmp_path: pathlib.Path) -> None:
    value = {"value": False}
    args = [value]
    trigger = FunctionTrigger(
        fn=lambda x: x["value"], args=args, trigger=(1, "iteration")
    )
    manager = ExtensionsManager(
        {}, {}, 1, iters_per_epoch=10, out_dir=str(tmp_path)
    )
    with manager.run_iteration():
        pass
    assert not trigger(manager)
    value["value"] = True
    assert trigger(manager)


@pytest.mark.parametrize(
    "trigger, iters_per_epoch",
    [((1, "iteration"), 10), ((1, "epoch"), 20), ((0.123, "epoch"), 17)],
)
def test_with_interval_trigger(
    trigger: TriggerLike, iters_per_epoch: int, tmp_path: pathlib.Path
) -> None:
    trigger = trigger_module.get_trigger(trigger)
    manager = ExtensionsManager(
        {}, [], 10, iters_per_epoch=iters_per_epoch, out_dir=str(tmp_path)
    )
    function_trigger = FunctionTrigger(fn=lambda: True, trigger=trigger)

    while not manager.stop_trigger:
        with manager.run_iteration():
            pass
        assert trigger(manager) == function_trigger(manager)
