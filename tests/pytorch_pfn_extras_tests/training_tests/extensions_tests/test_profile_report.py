import tempfile
import time
import os
import json

import pytest
import yaml

import pytorch_pfn_extras as ppe


def _body():
    with ppe.profiler.get_time_summary().report("iter-time"):
        time.sleep(0.1)


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
def test_profile_report(format, append):
    ext = ppe.training.extensions.ProfileReport(format=format, append=append)
    max_epochs = 3
    iters_per_epoch = 5
    # ppe.profiler.time_summary.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ppe.training.ExtensionsManager(
            {}, {}, max_epochs=max_epochs, iters_per_epoch=iters_per_epoch,
            out_dir=tmpdir)
        manager.extend(ext)
        for _epoch_idx in range(max_epochs):
            for _ in range(iters_per_epoch):
                with manager.run_iteration():
                    _body()
        with open(os.path.join(tmpdir, 'log')) as f:
            data = f.read()
            if format == 'json':
                values = json.loads(data)
            elif format == 'json-lines':
                values = [json.loads(x) for x in data.splitlines()]
            elif format == 'yaml':
                values = yaml.load(data, Loader=yaml.SafeLoader)
            assert len(values) == _epoch_idx + 1

            for value in values:
                assert abs(value['iter-time'] - 0.1) < 2e-2
