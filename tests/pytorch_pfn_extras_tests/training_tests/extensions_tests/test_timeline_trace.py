import json
import os
import tempfile
import time

import pytorch_pfn_extras as ppe


def _body():
    with ppe.profiler.record("tag", trace=True):
        time.sleep(0.1)


def test_tracer():
    max_epochs = 3
    iters_per_epoch = 5
    ppe.profiler.clear_tracer()
    ext = ppe.training.extensions.TimelineTrace(filename="trace.json")
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
