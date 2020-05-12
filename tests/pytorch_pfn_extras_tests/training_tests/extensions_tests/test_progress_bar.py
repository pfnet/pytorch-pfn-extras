import io

import pytorch_pfn_extras as ppe


def test_run():
    max_epochs = 10
    iters_per_epoch = 10
    manager = ppe.training.ExtensionsManager(
        {}, {}, max_epochs, iters_per_epoch=iters_per_epoch)

    out = io.StringIO()
    extension = ppe.training.extensions.ProgressBar(
        training_length=None,
        update_interval=1,
        bar_length=40,
        out=out,
    )
    manager.extend(extension)

    for epoch in range(max_epochs):
        for batch_idx in range(iters_per_epoch):
            with manager.run_iteration():
                if manager.updater.iteration < 2:
                    continue
                status = '{} iter, {} epoch / {} epochs'.format(
                    manager.updater.iteration, epoch, max_epochs)
                assert status in out.getvalue()
