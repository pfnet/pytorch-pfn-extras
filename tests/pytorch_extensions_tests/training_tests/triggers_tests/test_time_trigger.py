import pytorch_extensions as pe


class DummyTrainer:

    def __init__(self):
        self.elapsed_time = 0


def test_call():
    trigger = pe.training.triggers.TimeTrigger(1)
    trainer = DummyTrainer()

    assert not trigger(trainer)
    trainer.elapsed_time = 0.9
    assert not trigger(trainer)

    # first event is triggerred on time==1.0
    trainer.elapsed_time = 1.2
    assert trigger(trainer)

    trainer.elapsed_time = 1.3
    assert not trigger(trainer)

    # second event is triggerred on time==2.0, and is not on time==2.2
    trainer.elapsed_time = 2.1
    assert trigger(trainer)
