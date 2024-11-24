import os
import tempfile
import threading
import time
from contextlib import nullcontext

import pytest
import pytorch_pfn_extras as ppe
import torch

_profiler_available = os.name != "nt"


@pytest.mark.skipif(not _profiler_available, reason="profiler is not available")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_record(device):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = torch.nn.Linear(30, 40)
    model.to(device)
    x = torch.arange(30, dtype=torch.float32).to(device)

    with torch.profiler.profile() as prof:
        with ppe.profiler.record("my_tag_1"):
            model(x)

    keys = [event.key for event in prof.key_averages()]
    assert "my_tag_1" in keys
    assert "aten::linear" in keys


@pytest.mark.skipif(not _profiler_available, reason="profiler is not available")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_record_without_tag(device):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = torch.nn.Linear(30, 40)
    model.to(device)
    x = torch.arange(30, dtype=torch.float32).to(device)

    with torch.profiler.profile() as prof:
        with ppe.profiler.record(None):
            model(x)

    keys = [event.key for event in prof.key_averages()]
    assert "aten::linear" in keys
    assert any(k.endswith("test_record_without_tag") for k in keys)


@pytest.mark.skipif(not _profiler_available, reason="profiler is not available")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_record_function(device):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = torch.nn.Linear(30, 40)
    model.to(device)

    @ppe.profiler.record_function("my_tag_2")
    def my_run(x):
        model(x)

    with torch.profiler.profile() as prof:
        x = torch.arange(30, dtype=torch.float32).to(device)
        my_run(x)

    keys = [event.key for event in prof.key_averages()]
    assert "aten::linear" in keys
    assert "my_tag_2" in keys


@pytest.mark.skipif(not _profiler_available, reason="profiler is not available")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_record_function_without_tag(device):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = torch.nn.Linear(30, 40)
    model.to(device)
    x = torch.arange(30, dtype=torch.float32).to(device)

    @ppe.profiler.record_function(None)
    def my_run(x):
        model(x)

    with torch.profiler.profile() as prof:
        my_run(x)

    keys = [event.key for event in prof.key_averages()]
    assert "aten::linear" in keys
    assert "my_run" in keys


@pytest.mark.skipif(not _profiler_available, reason="profiler is not available")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_record_iterable(device):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = torch.nn.Linear(30, 40)
    model.to(device)

    x = torch.arange(30, dtype=torch.float32).to(device)
    iters = [x, x, x]

    with torch.profiler.profile() as prof:
        for x in ppe.profiler.record_iterable("my_tag_3", iters):
            model(x)

    keys = [event.key for event in prof.key_averages()]
    assert "aten::linear" in keys
    assert "my_tag_3-0" in keys
    assert "my_tag_3-1" in keys
    assert "my_tag_3-2" in keys


@pytest.mark.skipif(not _profiler_available, reason="profiler is not available")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_record_iterable_without_tag(device):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = torch.nn.Linear(30, 40)
    model.to(device)

    x = torch.arange(30, dtype=torch.float32).to(device)
    iters = [x, x, x]

    with torch.profiler.profile() as prof:
        for x in ppe.profiler.record_iterable(None, iters):
            model(x)

    keys = [event.key for event in prof.key_averages()]
    assert "aten::linear" in keys
    assert any(k.endswith("test_record_iterable_without_tag-0") for k in keys)
    assert any(k.endswith("test_record_iterable_without_tag-1") for k in keys)
    assert any(k.endswith("test_record_iterable_without_tag-2") for k in keys)


@pytest.mark.skipif(not _profiler_available, reason="profiler is not available")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_record_iterable_with_trace(device):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = torch.nn.Linear(30, 40)
    model.to(device)
    ppe.profiler.clear_tracer()

    x = torch.arange(30, dtype=torch.float32).to(device)

    with tempfile.TemporaryDirectory() as t_path:
        w = ppe.writing.SimpleWriter(out_dir=t_path)
        with torch.profiler.profile():
            with ppe.profiler.record("tag", trace=True):
                model(x)
        ppe.profiler.get_tracer().flush("trace.json", w)
        assert os.path.exists(os.path.join(t_path, "trace.json"))


@pytest.mark.skipif(not _profiler_available, reason="profiler is not available")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_record_iterable_with_threads(device):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = torch.nn.Linear(30, 40)
    model.to(device)
    ppe.profiler.clear_tracer()

    x = torch.arange(30, dtype=torch.float32).to(device)

    with torch.profiler.profile():

        def thread_body(thread_id):
            if device == "cuda":
                stream = torch.cuda.stream(torch.cuda.Stream())
            else:
                stream = nullcontext()
            for _ in range(10):
                with ppe.profiler.record(f"{thread_id}", trace=True), stream:
                    model(x)
                    # yield
                    time.sleep(0.0001)

        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=thread_body, args=(i,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert ppe.profiler.get_tracer()._event_count == 100
    with tempfile.TemporaryDirectory() as t_path:
        w = ppe.writing.SimpleWriter(out_dir=t_path)
        ppe.profiler.get_tracer().flush("trace.json", w)
        assert os.path.exists(os.path.join(t_path, "trace.json"))


@pytest.mark.skipif(not _profiler_available, reason="profiler is not available")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_record_iterable_with_thread_disabled(device):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = torch.nn.Linear(30, 40)
    model.to(device)
    ppe.profiler.clear_tracer()

    x = torch.arange(30, dtype=torch.float32).to(device)

    with torch.profiler.profile():

        def thread_body(thread_id):
            if device == "cuda":
                stream = torch.cuda.stream(torch.cuda.Stream())
            else:
                stream = nullcontext()
            try:
                if thread_id == 0:
                    ppe.profiler.enable_thread_trace(False)
                for _ in range(10):
                    with (
                        ppe.profiler.record(f"{thread_id}", trace=True),
                        stream,
                    ):
                        model(x)
                        # yield
                        time.sleep(0.0001)
            finally:
                ppe.profiler.enable_thread_trace(False)

        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=thread_body, args=(i,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert ppe.profiler.get_tracer()._event_count == 90
    with tempfile.TemporaryDirectory() as t_path:
        w = ppe.writing.SimpleWriter(out_dir=t_path)
        ppe.profiler.get_tracer().flush("trace.json", w)
        assert os.path.exists(os.path.join(t_path, "trace.json"))


@pytest.mark.skipif(not _profiler_available, reason="profiler is not available")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_record_iterable_with_all_thread_disabled(device):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = torch.nn.Linear(30, 40)
    model.to(device)
    ppe.profiler.clear_tracer()

    x = torch.arange(30, dtype=torch.float32).to(device)

    with torch.profiler.profile():

        def thread_body(thread_id):
            if device == "cuda":
                stream = torch.cuda.stream(torch.cuda.Stream())
            else:
                stream = nullcontext()
            for _ in range(10):
                with ppe.profiler.record(f"{thread_id}", trace=True), stream:
                    model(x)
                    # yield
                    time.sleep(0.0001)

        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=thread_body, args=(i,)))
        try:
            ppe.profiler.enable_global_trace(False)
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            ppe.profiler.enable_global_trace(True)
    assert ppe.profiler.get_tracer()._event_count == 0
    with tempfile.TemporaryDirectory() as t_path:
        w = ppe.writing.SimpleWriter(out_dir=t_path)
        ppe.profiler.get_tracer().flush("trace.json", w)
        assert os.path.exists(os.path.join(t_path, "trace.json"))


@pytest.mark.skipif(not _profiler_available, reason="profiler is not available")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_record_iterable_with_multiprocessing(device):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = torch.nn.Linear(30, 40)
    model.to(device)
    ppe.profiler.clear_tracer()

    data = [torch.arange(30, dtype=torch.float32)] * 5
    dataset = ppe.profiler.TraceableDataset(data, "tag")
    dataloader = ppe.dataloaders.DataLoader(dataset, num_workers=1)

    with tempfile.TemporaryDirectory() as t_path:
        w = ppe.writing.SimpleWriter(out_dir=t_path)
        ppe.profiler.get_tracer().initialize_writer("trace.json", w)
        with torch.profiler.profile():
            for x in dataloader:
                model(x.to(device))
        ppe.profiler.get_tracer().flush("trace.json", w)
        assert os.path.exists(os.path.join(t_path, "trace.json"))
        pid = os.getpid()
        values = ppe.profiler.load_chrome_trace_as_json(
            os.path.join(t_path, "trace.json")
        )
        assert len(values) == 5
        # Check that the values were written by a dataloader worker
        for v in values:
            assert v["pid"] != pid
