import io
import pathlib

import pytest
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training.extensions import _ipython_module_available
from pytorch_pfn_extras.training.extensions.log_report import _pandas_available


@pytest.mark.skipif(
    not _ipython_module_available or not _pandas_available,
    reason="print report notebook import failed, "
    "maybe ipython is not installed",
)
def test_run_print_report_notebook(tmp_path: pathlib.Path):
    max_epochs = 5
    iters_per_epoch = 5
    manager = ppe.training.ExtensionsManager(
        {},
        {},
        max_epochs,
        iters_per_epoch=iters_per_epoch,
        out_dir=str(tmp_path),
    )

    out = io.StringIO()
    log_report = ppe.training.extensions.LogReport()
    manager.extend(log_report)
    extension = ppe.training.extensions.PrintReportNotebook(out=out)
    manager.extend(extension)

    for _ in range(max_epochs):
        for _ in range(iters_per_epoch):
            with manager.run_iteration():
                # Only test it runs without fail
                # The value is not tested now...
                pass


@pytest.mark.skipif(
    not _ipython_module_available or not _pandas_available,
    reason="print report notebook import failed, "
    "maybe ipython is not installed",
)
def test_run_print_report_notebook_with_entry(tmp_path: pathlib.Path):
    max_epochs = 5
    iters_per_epoch = 5
    manager = ppe.training.ExtensionsManager(
        {},
        {},
        max_epochs,
        iters_per_epoch=iters_per_epoch,
        out_dir=str(tmp_path),
    )

    out = io.StringIO()
    log_report = ppe.training.extensions.LogReport()
    manager.extend(log_report)
    extension = ppe.training.extensions.PrintReportNotebook(
        ["epoch", "iteration"], out=out
    )
    manager.extend(extension)

    for _ in range(max_epochs):
        for _ in range(iters_per_epoch):
            with manager.run_iteration():
                # Only test it runs without fail
                # The value is not tested now...
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
