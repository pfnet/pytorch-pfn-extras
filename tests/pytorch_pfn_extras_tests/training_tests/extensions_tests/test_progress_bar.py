import io
import re
import time

import pytorch_pfn_extras as ppe


def test_run():
    max_epochs = 5
    iters_per_epoch = 5
    manager = ppe.training.ExtensionsManager(
        {}, {}, max_epochs, iters_per_epoch=iters_per_epoch
    )

    out = io.StringIO()
    extension = ppe.training.extensions.ProgressBar(
        training_length=None,
        update_interval=1,
        bar_length=40,
        out=out,
    )
    manager.extend(extension)

    for epoch in range(max_epochs):
        for _ in range(iters_per_epoch):
            with manager.run_iteration():
                time.sleep(0.1)
                if manager.iteration < 2:
                    continue
                status = out.getvalue()
                assert (
                    "{} iter, {} epoch / {} epochs".format(
                        manager.iteration, epoch, max_epochs
                    )
                    in status
                )
                iters_per_sec = float(
                    re.findall(r"([0-9]+\.[0-9]*) iters/sec", status)[-1]
                )
                assert 7 <= iters_per_sec <= 12
