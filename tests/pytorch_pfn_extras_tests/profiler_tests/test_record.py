import os
import tempfile

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
def test_record_iterable_with_chrome_trace(device):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip()
    model = torch.nn.Linear(30, 40)
    model.to(device)

    x = torch.arange(30, dtype=torch.float32).to(device)

    with tempfile.TemporaryDirectory() as t_path:
        w = ppe.writing.SimpleWriter(out_dir=t_path)
        with torch.profiler.profile():
            with ppe.profiler.record("tag", trace=True):
                model(x)
        ppe.profiler.get_tracer().flush("trace.json", w)
        assert os.path.exists(os.path.join(t_path, "trace.json"))
