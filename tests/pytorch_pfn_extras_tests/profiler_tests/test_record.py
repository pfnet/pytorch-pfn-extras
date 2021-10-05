import torch

import pytorch_pfn_extras as ppe


def test_record():
    model = torch.nn.Linear(30, 40)
    x = torch.arange(30, dtype=torch.float32)

    with torch.profiler.profile() as prof:
        with ppe.profiler.record('my_tag_1'):
            model(x)

    keys = [event.key for event in prof.key_averages()]
    assert 'my_tag_1' in keys
    assert 'aten::linear' in keys


def test_record_without_tag():
    model = torch.nn.Linear(30, 40)
    x = torch.arange(30, dtype=torch.float32)

    with torch.profiler.profile() as prof:
        with ppe.profiler.record(None):
            model(x)

    keys = [event.key for event in prof.key_averages()]
    assert 'aten::linear' in keys
    assert any(k.endswith('test_record_without_tag') for k in keys)


def test_record_function():
    model = torch.nn.Linear(30, 40)

    @ppe.profiler.record_function('my_tag_2')
    def my_run(x):
        model(x)

    with torch.profiler.profile() as prof:
        x = torch.arange(30, dtype=torch.float32)
        my_run(x)

    keys = [event.key for event in prof.key_averages()]
    assert 'aten::linear' in keys
    assert 'my_tag_2' in keys


def test_record_function_without_tag():
    model = torch.nn.Linear(30, 40)
    x = torch.arange(30, dtype=torch.float32)

    @ppe.profiler.record_function(None)
    def my_run(x):
        model(x)

    with torch.profiler.profile() as prof:
        my_run(x)

    keys = [event.key for event in prof.key_averages()]
    assert 'aten::linear' in keys
    assert 'my_run' in keys


def test_record_iterable():
    model = torch.nn.Linear(30, 40)

    x = torch.arange(30, dtype=torch.float32)
    iters = [x, x, x]

    with torch.profiler.profile() as prof:
        for x in ppe.profiler.record_iterable('my_tag_3', iters):
            model(x)

    keys = [event.key for event in prof.key_averages()]
    assert 'aten::linear' in keys
    assert 'my_tag_3-0' in keys
    assert 'my_tag_3-1' in keys
    assert 'my_tag_3-2' in keys


def test_record_iterable_without_tag():
    model = torch.nn.Linear(30, 40)

    x = torch.arange(30, dtype=torch.float32)
    iters = [x, x, x]

    with torch.profiler.profile() as prof:
        for x in ppe.profiler.record_iterable(None, iters):
            model(x)

    keys = [event.key for event in prof.key_averages()]
    assert 'aten::linear' in keys
    assert any(k.endswith('test_record_iterable_without_tag-0') for k in keys)
    assert any(k.endswith('test_record_iterable_without_tag-1') for k in keys)
    assert any(k.endswith('test_record_iterable_without_tag-2') for k in keys)
