from pytorch_pfn_extras.training import extension


class LRScheduler(extension.Extension):
    """Wraps a `torch.optim.lr_scheduler` extension
    so it will be called after the corresponding iteration.
    This allows to take snapshots and automatically step them.
    """
    def __init__(self, torch_lr_scheduler):
        self._lr_scheduler = torch_lr_scheduler

    def __call__(self, manager):
        self._lr_scheduler.step()

    def state_dict(self):
        return self._lr_scheduler.state_dict()

    def load_state_dict(self, state):
        self._lr_scheduler.load_state_dict(state)
