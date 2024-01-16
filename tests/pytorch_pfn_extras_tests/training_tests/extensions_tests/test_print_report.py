import io
import pathlib

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions


def test_run_print_report(tmp_path: pathlib.Path):
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
    log_report = extensions.LogReport()
    manager.extend(log_report)
    extension = extensions.PrintReport(out=out)
    manager.extend(extension)

    for _ in range(max_epochs):
        for _ in range(iters_per_epoch):
            with manager.run_iteration():
                pass
        assert "epoch       elapsed_time" in out.getvalue()
