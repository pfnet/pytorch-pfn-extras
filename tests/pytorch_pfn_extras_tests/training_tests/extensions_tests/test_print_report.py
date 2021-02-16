import io

import pytest

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions


@pytest.mark.parametrize(
    'format,append',
    [
        ('json', False),
        ('json-lines', True),
        ('json-lines', False),
        ('yaml', True),
        ('yaml', False),
    ]
)
def test_run_print_report(format, append):
    max_epochs = 5
    iters_per_epoch = 5
    manager = ppe.training.ExtensionsManager(
        {}, {}, max_epochs, iters_per_epoch=iters_per_epoch)

    out = io.StringIO()
    log_report = extensions.LogReport(format=format, append=append)
    manager.extend(log_report)
    extension = extensions.PrintReport(out=out)
    manager.extend(extension)

    for epoch in range(max_epochs):
        for batch_idx in range(iters_per_epoch):
            with manager.run_iteration():
                pass
        assert "epoch       elapsed_time" in out.getvalue()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
