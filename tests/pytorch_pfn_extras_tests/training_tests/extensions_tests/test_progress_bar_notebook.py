import io

import pytest
import torch
from torch import nn
from torch.nn import Linear
from torch.optim.adam import Adam

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras import training
from pytorch_pfn_extras.training.extensions import _ipython_module_available


@pytest.mark.skipif(
    not _ipython_module_available,
    reason="progress bar notebook import failed, "
           "maybe ipython is not installed"
)
def test_run_progress_bar_notebook():
    max_epochs = 5
    iters_per_epoch = 5
    manager = ppe.training.ExtensionsManager(
        {}, {}, max_epochs, iters_per_epoch=iters_per_epoch)

    out = io.StringIO()
    extension = ppe.training.extensions.ProgressBarNotebook(
        training_length=None,
        update_interval=1,
        bar_length=40,
        out=out,
    )
    manager.extend(extension)

    for epoch in range(max_epochs):
        for _ in range(iters_per_epoch):
            with manager.run_iteration():
                if manager.iteration < 2:
                    continue
                status = '{} iter, {} epoch / {} epochs'.format(
                    manager.iteration, epoch, max_epochs)
                assert status in extension._status_html.value


@pytest.mark.skipif(
    not _ipython_module_available,
    reason="progress bar notebook import failed, "
           "maybe ipython is not installed"
)
def test_ignite_extensions_manager_with_progressbar_notebook():

    try:
        from ignite.engine import create_supervised_trainer
    except ImportError:
        pytest.skip('pytorch-ignite not found')

    max_epochs = 5
    iters_per_epoch = 4

    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.lin = Linear(1, 1)

        def forward(self, *args):
            pass

    model = DummyModel()
    optimizer = Adam(model.parameters())

    def _fake_loss(*args):
        return torch.tensor([0.0], requires_grad=True)

    trainer = create_supervised_trainer(
        model, optimizer, _fake_loss)

    manager = training.IgniteExtensionsManager(
        trainer,
        {'model_name': model},
        {'optimizer_name': optimizer},
        max_epochs,
    )
    manager.extend(ppe.training.extensions.ProgressBarNotebook())

    loader = torch.utils.data.DataLoader(
        [(i, i) for i in range(iters_per_epoch)])
    trainer.run(loader, max_epochs=max_epochs)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
