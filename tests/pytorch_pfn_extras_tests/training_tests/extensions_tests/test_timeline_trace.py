import os
import tempfile
import time

import pytorch_pfn_extras as ppe


def test_tracer_tls():
    def _body():
        with ppe.profiler.record("tag", trace=True):
            time.sleep(0.1)

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
        values = ppe.profiler.load_chrome_trace_as_json(
            os.path.join(tmpdir, "trace.json")
        )
        assert len(values) == max_epochs * iters_per_epoch
    ppe.profiler.clear_tracer()


def test_tracer_object():
    max_epochs = 3
    iters_per_epoch = 5
    tracer = ppe.profiler.ChromeTracer()
    ext = ppe.training.extensions.TimelineTrace(
        filename="trace.json", tracer=tracer
    )

    def _body():
        with ppe.profiler.record("tag", trace=tracer):
            time.sleep(0.1)

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ppe.training.ExtensionsManager(
            {},
            {},
            max_epochs=max_epochs,
            iters_per_epoch=iters_per_epoch,
            out_dir=tmpdir,
        )
        manager.extend(ext)
        count = 0
        for _epoch_idx in range(max_epochs):
            for _ in range(iters_per_epoch):
                with manager.run_iteration():
                    _body()
                count += 1
                assert tracer._event_count == count
        values = ppe.profiler.load_chrome_trace_as_json(
            os.path.join(tmpdir, "trace.json")
        )
        assert len(values) == max_epochs * iters_per_epoch


def test_tracer_no_append_object():
    max_epochs = 3
    iters_per_epoch = 5
    tracer = ppe.profiler.ChromeTracer(append=False)
    ext = ppe.training.extensions.TimelineTrace(
        filename="trace.json", tracer=tracer
    )

    def _body():
        with ppe.profiler.record("tag", trace=tracer):
            time.sleep(0.1)

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ppe.training.ExtensionsManager(
            {},
            {},
            max_epochs=max_epochs,
            iters_per_epoch=iters_per_epoch,
            out_dir=tmpdir,
        )
        manager.extend(ext)
        count = 0
        for _epoch_idx in range(max_epochs):
            for _ in range(iters_per_epoch):
                with manager.run_iteration():
                    _body()
                count += 1
                assert tracer._event_count == count
        values = ppe.profiler.load_chrome_trace_as_json(
            os.path.join(tmpdir, "trace.json")
        )
        assert len(values) == max_epochs * iters_per_epoch


def test_tracer_enable():
    max_epochs = 4
    iters_per_epoch = 5
    tracer = ppe.profiler.ChromeTracer()
    disable = ppe.training.triggers.ManualScheduleTrigger(
        [10], unit="iteration"
    )
    enable = ppe.training.triggers.ManualScheduleTrigger([15], unit="iteration")
    ext = ppe.training.extensions.TimelineTrace(
        filename="trace.json", tracer=tracer, enable=enable, disable=disable
    )

    def _body():
        with ppe.profiler.record("tag", trace=tracer):
            time.sleep(0.1)

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ppe.training.ExtensionsManager(
            {},
            {},
            max_epochs=max_epochs,
            iters_per_epoch=iters_per_epoch,
            out_dir=tmpdir,
        )
        manager.extend(ext)
        count = 0
        for _epoch_idx in range(max_epochs):
            for _ in range(iters_per_epoch):
                with manager.run_iteration():
                    _body()
                count += 1
        values = ppe.profiler.load_chrome_trace_as_json(
            os.path.join(tmpdir, "trace.json")
        )
        assert len(values) == (max_epochs * iters_per_epoch - 10 + 5)
