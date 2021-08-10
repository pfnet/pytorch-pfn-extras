import pytest
import pytorch_pfn_extras as ppe


class TestEngine:
    def test_engine_extension(self):
        engine = ppe.engine._Engine(
            None, {}, optimizers={}, max_epochs=10)
        extension = ppe.training.extensions.LogReport()
        engine.extend(extension)
        # Create the actual manager object
        engine._setup_manager(10)
        # Extension is not stored as is so we make sure it was detected
        assert len(engine.manager._extensions) == 1

    def test_engine_state_dict(self):
        manager = ppe.training.ExtensionsManager(
            {}, {}, 10, iters_per_epoch=100)
        engine = ppe.engine._Engine(
            None, {}, optimizers={}, max_epochs=10)
        engine._setup_manager(100)
        assert engine.state_dict() == manager.state_dict()

    def test_engine_load_state_dict(self):
        manager = ppe.training.ExtensionsManager(
            {}, {}, 10, iters_per_epoch=100)
        engine = ppe.engine._Engine(None, {}, optimizers={}, max_epochs=1)
        engine.load_state_dict(manager.state_dict())
        engine._setup_manager(20)
        assert engine.state_dict() == manager.state_dict()

    def test_engine_load_state_dict_2(self):
        manager = ppe.training.ExtensionsManager(
            {}, {}, 10, iters_per_epoch=100)
        engine = ppe.engine._Engine(None, {}, optimizers={}, max_epochs=1)
        engine._setup_manager(20)
        engine.load_state_dict(manager.state_dict())
        assert engine.state_dict() == manager.state_dict()


class TestEngineInvalid:
    def test_engine_wrong_models(self):
        with pytest.raises(ValueError, match='model must be an instance'):
            ppe.engine._Engine(None, object(), optimizers={}, max_epochs=10)

    def test_engine_not_started(self):
        engine = ppe.engine._Engine(None, {}, optimizers={}, max_epochs=10)
        with pytest.raises(RuntimeError, match='is not started'):
            engine.state_dict()
        with pytest.raises(RuntimeError, match='is not started'):
            engine.manager

    def test_extend_after_init(self):
        engine = ppe.engine._Engine(
            None, {}, optimizers={}, max_epochs=10)
        engine._setup_manager(10)
        extension = ppe.training.extensions.LogReport()
        with pytest.raises(RuntimeError, match='cannot extend after'):
            engine.extend(extension)
