import io
import os
import tempfile
import yaml

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions
from pytorch_pfn_extras.training.extensions import log_report as log_report_module


def test_log_buffer():
    buf = log_report_module._LogBuffer()
    looker = buf.emit_new_looker()
    assert buf.size() == 0
    buf.append('mes1')
    buf.append('mes2')
    assert buf.size() == 2
    assert looker.get() == ['mes1', 'mes2']
    assert buf.size() == 2
    looker.clear()
    assert buf.size() == 0
    assert looker.get() == []
    buf.append('mes3')
    assert buf.size() == 1
    assert looker.get() == ['mes3']
    assert buf.size() == 1
    looker.clear()
    assert buf.size() == 0
    assert looker.get() == []


def test_log_buffer_multiple_lookers():
    buf = log_report_module._LogBuffer()
    looker1 = buf.emit_new_looker()
    looker2 = buf.emit_new_looker()
    buf.append('mes1')
    assert looker1.get() == ['mes1']
    assert looker2.get() == ['mes1']
    assert buf.size() == 1
    looker2.clear()
    assert buf.size() == 1
    buf.append('mes2')
    assert looker1.get() == ['mes1', 'mes2']
    assert looker2.get() == ['mes2']
    assert buf.size() == 2
    looker2.clear()
    assert buf.size() == 2
    looker1.clear()
    assert buf.size() == 0


def test_buffer_size_log_report():
    max_epochs = 10
    iters_per_epoch = 5

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ppe.training.ExtensionsManager(
            {}, {}, max_epochs, iters_per_epoch=iters_per_epoch, out_dir=tmpdir)

        log_report = extensions.LogReport(
            filename='out', format='yaml', append=True)
        manager.extend(log_report, (1, 'epoch'))

        for _ in range(max_epochs):
            for _ in range(iters_per_epoch):
                assert log_report._log_buffer.size() <= 1
                with manager.run_iteration():
                    pass

        with open(os.path.join(tmpdir, 'out')) as f:
            data = f.read()
            values = yaml.load(data, Loader=yaml.SafeLoader)
            assert len(values) == max_epochs


def test_buffer_size_log_report_and_print_report():
    max_epochs = 10
    iters_per_epoch = 5

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ppe.training.ExtensionsManager(
            {}, {}, max_epochs, iters_per_epoch=iters_per_epoch, out_dir=tmpdir)

        log_report = extensions.LogReport(
            filename='out', format='yaml', append=True)
        manager.extend(log_report, trigger=(1, 'epoch'))

        out = io.StringIO()
        print_report = extensions.PrintReport(out=out)
        manager.extend(print_report, trigger=(3, 'epoch'))

        for _ in range(max_epochs):
            for _ in range(iters_per_epoch):
                assert log_report._log_buffer.size() <= 3
                with manager.run_iteration():
                    pass

        with open(os.path.join(tmpdir, 'out')) as f:
            data = f.read()
            values = yaml.load(data, Loader=yaml.SafeLoader)
            assert len(values) == max_epochs
