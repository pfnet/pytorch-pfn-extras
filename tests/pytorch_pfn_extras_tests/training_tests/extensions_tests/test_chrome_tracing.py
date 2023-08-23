import json
import os
import tempfile
import time

import pytest
import pytorch_pfn_extras as ppe


def _body():
    with ppe.profiler.record("tag", emit_chrome_trace=True):
        time.sleep(0.1)


@pytest.mark.parametrize(
    "format,append",
    [
        ("json", False),
        ("json-lines", True),
        ("json-lines", False),
        ("yaml", True),
        ("yaml", False),
    ],
)
def test_profile_report(format, append):
    ext = ppe.training.extensions.ChromeTrace(filename="trace.json")
    max_epochs = 3
    iters_per_epoch = 5
    # ppe.profiler.time_summary.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ppe.training.ExtensionsManager(
            {},
            {},
            max_epochs=max_epochs,
            iters_per_epoch=iters_per_epoch,
            out_dir=tmpdir,
        )
        manager.extend(ext)
        for _epoch_idx in range(max_epochs):
            for _ in range(iters_per_epoch):
                with manager.run_iteration():
                    _body()
        with open(os.path.join(tmpdir, "trace.json")) as f:
            data = f.read()
            values = json.loads(data)
            assert len(values) == max_epochs * iters_per_epoch
