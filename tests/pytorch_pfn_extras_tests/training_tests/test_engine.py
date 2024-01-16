import pathlib
from typing import Any, Dict

import pytest
import pytorch_pfn_extras as ppe


class TestEngine:
    def test_engine_extension(self, tmp_path: pathlib.Path):
        engine = ppe.training._trainer.Trainer(
            None,
            evaluator=None,
            models={},
            optimizers={},
            max_epochs=10,
            out_dir=str(tmp_path),
        )
        extension = ppe.training.extensions.LogReport()
        engine.extend(extension)
        # Create the actual manager object
        engine._setup_manager(10)
        # Extension is not stored as is so we make sure it was detected
        assert len(engine.manager._extensions) == 1

    def test_engine_state_dict(self, tmp_path: pathlib.Path):
        manager = ppe.training.ExtensionsManager(
            {},
            {},
            10,
            iters_per_epoch=100,
            out_dir=str(tmp_path),
        )
        engine = ppe.training._trainer.Trainer(
            None,
            evaluator=None,
            models={},
            optimizers={},
            max_epochs=10,
            out_dir=str(tmp_path),
        )
        engine._setup_manager(100)
        assert engine.state_dict() == manager.state_dict()

    def test_engine_load_state_dict(self, tmp_path: pathlib.Path):
        manager = ppe.training.ExtensionsManager(
            {}, {}, 10, iters_per_epoch=100, out_dir=str(tmp_path)
        )
        engine = ppe.training._trainer.Trainer(
            None,
            evaluator=None,
            models={},
            optimizers={},
            max_epochs=1,
            out_dir=str(tmp_path),
        )
        engine.load_state_dict(manager.state_dict())
        engine._setup_manager(20)
        assert engine.state_dict() == manager.state_dict()

    def test_engine_load_state_dict_2(self, tmp_path: pathlib.Path):
        manager = ppe.training.ExtensionsManager(
            {},
            {},
            10,
            iters_per_epoch=100,
            out_dir=str(tmp_path),
        )
        engine = ppe.training._trainer.Trainer(
            None,
            evaluator=None,
            models={},
            optimizers={},
            max_epochs=1,
            out_dir=str(tmp_path),
        )
        engine._setup_manager(20)
        engine.load_state_dict(manager.state_dict())
        assert engine.state_dict() == manager.state_dict()


class TestEngineInvalid:
    def test_engine_wrong_models(self):
        with pytest.raises(ValueError, match="model must be an instance"):
            ppe.training._trainer.Trainer(
                None,
                evaluator=None,
                models=object(),
                optimizers={},
                max_epochs=10,
            )

    def test_engine_not_started(self):
        engine = ppe.training._trainer.Trainer(
            None, evaluator=None, models={}, optimizers={}, max_epochs=10
        )
        with pytest.raises(RuntimeError, match="is not started"):
            engine.state_dict()
        with pytest.raises(RuntimeError, match="is not started"):
            engine.manager

    def test_extend_after_init(self, tmp_path: pathlib.Path):
        engine = ppe.training._trainer.Trainer(
            None,
            evaluator=None,
            models={},
            optimizers={},
            max_epochs=10,
            out_dir=str(tmp_path),
        )
        engine._setup_manager(10)
        extension = ppe.training.extensions.LogReport()
        with pytest.raises(RuntimeError, match="cannot extend after"):
            engine.extend(extension)


class DummyStateObjects:
    def __init__(self) -> None:
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        return


@pytest.mark.parametrize(
    "args",
    [
        {"a.b": DummyStateObjects(), "a": {"b": DummyStateObjects()}},
        {"a": DummyStateObjects(), "b": DummyStateObjects()},
        {
            "a__.__b": DummyStateObjects(),
            "__a__": {"__b__": DummyStateObjects()},
        },
        {"a::b": DummyStateObjects(), "a": {"b": DummyStateObjects()}},
        {"a:b": DummyStateObjects(), "a": {"b": DummyStateObjects()}},
    ],
)
def test_filter_state_objects(args) -> None:
    out = ppe.engine.filter_state_objects(args)
    key_set = set()
    for key, _ in out:
        assert key not in key_set
        key_set.add(key)
