from typing import Dict
from unittest.mock import MagicMock

import pytest
from pytorch_pfn_extras.training import ExtensionsManager
from pytorch_pfn_extras.training.extensions.call_function import CallFunction


def test_function_is_called() -> None:
    fn = MagicMock()
    args = [MagicMock()]
    kwargs = {"a": MagicMock()}

    extension = CallFunction(fn=fn, args=args, kwargs=kwargs)
    fn.assert_not_called()
    extension(MagicMock())
    fn.assert_called_once_with(*args, **kwargs)


def test_report() -> None:
    def add_fn(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
        (a_val,) = list(a.values())
        (b_val,) = list(b.values())
        return {"ret": a_val + b_val}

    a = {"0": 0}
    b = {"0": 0}

    extension = CallFunction(fn=add_fn, args=[a, b])

    epoch = 1
    iteration = 10
    manager = ExtensionsManager({}, {}, epoch, iters_per_epoch=iteration)
    manager.extend(extension, trigger=(1, "iteration"))
    a_val = 0
    b_val = 0
    while not manager.stop_trigger:
        a["0"] = a_val
        b["0"] = b_val
        with manager.run_iteration():
            pass
        assert manager.observation["ret"] == a_val + b_val
        a_val += 1
        b_val += 1


@pytest.mark.parametrize("report_keys", [["ret"], ["other"], ["ret", "other"]])
def test_report_with_key(report_keys) -> None:
    def add_fn(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
        (a_val,) = list(a.values())
        (b_val,) = list(b.values())
        return {
            "ret": a_val + b_val,
            "other": 0,
        }

    a = {"0": 0}
    b = {"0": 0}

    extension = CallFunction(fn=add_fn, args=[a, b], report_keys=report_keys)

    epoch = 1
    iteration = 10
    manager = ExtensionsManager({}, {}, epoch, iters_per_epoch=iteration)
    manager.extend(extension, trigger=(1, "iteration"))
    while not manager.stop_trigger:
        with manager.run_iteration():
            pass
        assert set(manager.observation.keys()) == set(report_keys)


def test_report_with_prefix() -> None:
    def add_fn(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
        (a_val,) = list(a.values())
        (b_val,) = list(b.values())
        return {"ret": a_val + b_val}

    a = {"0": 0}
    b = {"0": 0}

    extension = CallFunction(fn=add_fn, args=[a, b], report_prefix="prefix")
    expected_keys = set(["prefix/ret"])

    epoch = 1
    iteration = 10
    manager = ExtensionsManager({}, {}, epoch, iters_per_epoch=iteration)
    manager.extend(extension, trigger=(1, "iteration"))
    while not manager.stop_trigger:
        with manager.run_iteration():
            pass
        assert set(manager.observation.keys()) == expected_keys


@pytest.mark.parametrize("run_on_error", [True, False])
def test_on_error(run_on_error) -> None:
    fn = MagicMock()
    args = [MagicMock()]
    kwargs = {"a": MagicMock()}

    extension = CallFunction(
        fn=fn, args=args, kwargs=kwargs, run_on_error=run_on_error
    )
    epoch = 1
    iteration = 10
    manager = ExtensionsManager({}, {}, epoch, iters_per_epoch=iteration)
    manager.extend(extension, trigger=(1000, "iteration"))
    fn.assert_not_called()
    with manager.run_iteration():
        pass
    fn.assert_not_called()

    try:
        with manager.run_iteration():
            raise RuntimeError
    except RuntimeError:
        if run_on_error:
            fn.assert_called_once_with(*args, **kwargs)
        else:
            fn.assert_not_called()
