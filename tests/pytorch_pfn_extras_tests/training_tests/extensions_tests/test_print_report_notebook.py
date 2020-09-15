import io

import pytest

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training.extensions import _ipython_module_available


@pytest.mark.skipif(
    not _ipython_module_available,
    reason="print report notebook import failed, "
           "maybe ipython is not installed"
)
def test_run_progress_bar_notebook():
    max_epochs = 5
    iters_per_epoch = 5
    manager = ppe.training.ExtensionsManager(
        {}, {}, max_epochs, iters_per_epoch=iters_per_epoch)

    out = io.StringIO()
    log_report = ppe.training.extensions.LogReport()
    manager.extend(log_report)
    extension = ppe.training.extensions.PrintReportNotebook(out=out)
    manager.extend(extension)

    for epoch in range(max_epochs):
        for batch_idx in range(iters_per_epoch):
            with manager.run_iteration():
                if manager.iteration < 2:
                    continue
                # Only test it runs without fail
                # The value is not tested now...


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
