import io

import pytest

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training.extensions import _ipython_module_available


@pytest.mark.skipif(
    not _ipython_module_available,
    reason="progress bar notebook import failed, maybe ipython is not installed"
)
def test_run_progress_bar_notebook():
    max_epochs = 10
    iters_per_epoch = 10
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
        for batch_idx in range(iters_per_epoch):
            with manager.run_iteration():
                if manager.iteration < 2:
                    continue
                status = '{} iter, {} epoch / {} epochs'.format(
                    manager.iteration, epoch, max_epochs)
                assert status in extension._status_html.value


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
