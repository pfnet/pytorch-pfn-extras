import json
import os.path
import tempfile

import pytest
import yaml

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions


@pytest.mark.parametrize(
    'filename,expected_format',
    [
        ('out.json', 'json'),
        ('out.xyz', 'json'),
        (None, 'json'),
        ('out.yaml', 'yaml'),
        ('out.jsonl', 'json-lines'),
    ]
)
def test_format_from_ext(filename, expected_format):
    log_report = extensions.LogReport(filename=filename, format=None)
    assert log_report._format == expected_format


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
def test_output(format, append):
    max_epochs = 3
    iters_per_epoch = 5

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ppe.training.ExtensionsManager(
            {}, {}, max_epochs=max_epochs, iters_per_epoch=iters_per_epoch,
            out_dir=tmpdir)
        log_report = extensions.LogReport(
            filename='out', format=format, append=append)
        manager.extend(log_report)
        for epoch_idx in range(max_epochs):
            for _ in range(iters_per_epoch):
                with manager.run_iteration():
                    pass
            with open(os.path.join(tmpdir, 'out')) as f:
                data = f.read()
                if format == 'json':
                    values = json.loads(data)
                elif format == 'json-lines':
                    values = [json.loads(x) for x in data.splitlines()]
                elif format == 'yaml':
                    values = yaml.load(data, Loader=yaml.SafeLoader)
                assert len(values) == epoch_idx + 1
                this_epoch = values.pop()
                assert this_epoch['epoch'] == epoch_idx + 1
                assert (this_epoch['iteration']
                        == (epoch_idx + 1) * iters_per_epoch)
                assert 0 < this_epoch['elapsed_time']


def test_tensorboard_writer():
    pytest.importorskip('tensorboard')

    max_epochs = 3
    iters_per_epoch = 5

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = ppe.writing.TensorBoardWriter(out_dir=tmpdir)
        log_report = extensions.LogReport(
            writer=writer, trigger=(1, 'iteration'))
        manager = ppe.training.ExtensionsManager(
            {}, {}, max_epochs=max_epochs, iters_per_epoch=iters_per_epoch,
            out_dir=tmpdir)
        manager.extend(log_report)
        for _ in range(max_epochs):
            for _ in range(iters_per_epoch):
                with manager.run_iteration():
                    pass
        writer.finalize()

        files = os.listdir(tmpdir)
        assert len(files) == 1
        tb_file = files[0]
        assert tb_file.startswith('events.out.')

        # Won't play with protobuf, just ensure that our keys are in.
        with open(os.path.join(tmpdir, tb_file), 'rb') as f:
            tb_data = f.read()
        for key in ['epoch', 'iteration', 'elapsed_time']:
            assert key.encode('ascii') in tb_data


def test_deferred_values():
    max_epochs = 3
    iters_per_epoch = 5

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ppe.training.ExtensionsManager(
            {}, {}, max_epochs=max_epochs, iters_per_epoch=iters_per_epoch,
            out_dir=tmpdir)
        log_report = extensions.LogReport(filename="out")
        manager.extend(log_report)
        for epoch_idx in range(max_epochs):
            for _ in range(iters_per_epoch):
                with manager.run_iteration():
                    ppe.reporting.report({"x": lambda: epoch_idx})
            with open(os.path.join(tmpdir, 'out')) as f:
                data = f.read()
                values = json.loads(data)
                assert len(values) == epoch_idx + 1
                this_epoch = values.pop()
                assert this_epoch["x"] == epoch_idx
