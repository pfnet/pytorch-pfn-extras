import tempfile
import typing

import pytest
import torch
import torch.nn.functional as F

import pytorch_pfn_extras as ppe
from tests.pytorch_pfn_extras_tests.runtime_tests.test_jit_runtime import JITRuntime
from tests.pytorch_pfn_extras_tests.training_tests import test_trainer


class Model(torch.nn.Module):
    def __init__(self, device, ret_val):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(10))
        self.device = device
        self.ret_val = ret_val
        self.iter = 0

    def forward(self, x):
        a = torch.tensor(self.ret_val, device=self.device)
        a.requires_grad = True
        self.iter += 1
        return {"a": a, "iter": self.iter}


def _get_trainer(
        model_class, device, args, loader, *,
        seed=0, max_epochs=10, stop_trigger=None):
    torch.manual_seed(seed)
    model = model_class(device, *args)
    ppe.to(model, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    trainer = ppe.engine.create_trainer(
        model, optimizer, max_epochs, device=device, stop_trigger=stop_trigger)
    return trainer, (loader,)


def _get_evaluator(model_class, device, args, loader, *, seed=0, max_epochs=None):
    torch.manual_seed(seed)
    model = model_class(device, *args)
    ppe.to(model, device)
    evaluator = ppe.engine.create_evaluator(model, device=device)
    return evaluator, (loader,)


def _get_trainer_with_evaluator(
        model_class, device, args, loader, *,
        seed=0, max_epochs=10, stop_trigger=None):
    torch.manual_seed(seed)
    model = model_class(device, *args)
    ppe.to(model, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    evaluator = ppe.engine.create_evaluator(model, device=device)
    trainer = ppe.engine.create_trainer(
        model, optimizer, max_epochs, device=device,
        evaluator=evaluator, stop_trigger=stop_trigger)
    return trainer, (loader, loader)


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_compare_every_epoch(engine_fn):
    loader = list(torch.ones(10) for _ in range(10))
    engine_cpu, loaders_cpu = engine_fn(Model, "cpu", [1.0], loader)
    engine_gpu, loaders_gpu = engine_fn(Model, "cuda:0", [1.0], loader)
    comp = ppe.utils.comparer.Comparer(outputs=["a"])
    comp.add_engine("cpu", engine_cpu, *loaders_cpu)
    comp.add_engine("gpu", engine_gpu, *loaders_gpu)
    comp.compare()


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_comparer_wrong(engine_fn):
    loader = list(torch.ones(10) for _ in range(10))
    engine_cpu, loaders_cpu = engine_fn(Model, "cpu", [1.0], loader)
    engine_gpu, loaders_gpu = engine_fn(Model, "cuda:0", [0.5], loader)
    comp = ppe.utils.comparer.Comparer(outputs=["a"])
    comp.add_engine("cpu", engine_cpu, *loaders_cpu)
    comp.add_engine("gpu", engine_gpu, *loaders_gpu)
    with pytest.raises(AssertionError):
        comp.compare()


class _CustomComparer:
    def __init__(self, n_iters=None):
        self.times_called = 0
        self.n_iters = n_iters

    def __call__(self, eng_name_1, eng_name_2, out_name, out_1, out_2):
        assert out_name in ("output:a", "output:iter",)
        assert eng_name_1 in ("cpu", "gpu")
        assert eng_name_1 != eng_name_2
        if out_name == "output:iter":
            assert out_1 == out_2
            expected = [3, 6, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            assert out_1 == expected[self.times_called]
            self.times_called += 1
        else:
            assert out_1.cpu() == out_2.cpu()


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_trainer_with_evaluator])
def test_comparer_trigger(engine_fn):
    n_iters = 3
    loader = list(torch.ones(10) for _ in range(10))
    trainer_cpu, loaders_cpu = engine_fn(Model, "cpu", [1.0], loader, max_epochs=1)
    trainer_gpu, loaders_gpu = engine_fn(Model, "cuda:0", [1.0], loader, max_epochs=1)
    compare_fn = _CustomComparer(n_iters)
    comp = ppe.utils.comparer.Comparer(
        trigger=(n_iters, "iteration"), compare_fn=compare_fn)
    comp.add_engine("cpu", trainer_cpu, *loaders_cpu)
    comp.add_engine("gpu", trainer_gpu, *loaders_gpu)
    comp.compare()
    if engine_fn is _get_trainer_with_evaluator:
        assert compare_fn.times_called == 13
    else:
        assert compare_fn.times_called == 3


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_comparer_kwargs(engine_fn):
    loader = list(torch.ones(10) for _ in range(10))
    engine_cpu, loaders_cpu = engine_fn(Model, "cpu", [1.0], loader)
    engine_gpu, loaders_gpu = engine_fn(Model, "cuda:0", [0.991], loader)
    compare_fn = ppe.utils.comparer.get_default_comparer(rtol=1e-2, atol=1e-2)
    comp = ppe.utils.comparer.Comparer(outputs=["a"], compare_fn=compare_fn)
    comp.add_engine("cpu", engine_cpu, *loaders_cpu)
    comp.add_engine("gpu", engine_gpu, *loaders_gpu)
    comp.compare()


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_trainer_with_evaluator])
def test_comparer_incompat_trigger(engine_fn):
    loader = list(torch.ones(10) for _ in range(10))
    trainer_cpu, loaders_cpu = engine_fn(Model, "cpu", [1.0], loader)
    trainer_gpu, loaders_gpu = engine_fn(Model, "cuda:0", [1.0], loader,
                                         stop_trigger=(1, "iteration"))
    comp = ppe.utils.comparer.Comparer(outputs=["a"])
    comp.add_engine("cpu", trainer_cpu, *loaders_cpu)
    comp.add_engine("gpu", trainer_gpu, *loaders_gpu)
    with pytest.raises(ValueError):
        comp.compare()


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_compare_concurrency(engine_fn):
    loader = list(torch.ones(10) for _ in range(10))
    engine_cpu, loaders_cpu = engine_fn(Model, "cpu", [1.0], loader)
    engine_gpu, loaders_gpu = engine_fn(Model, "cuda:0", [1.0], loader)
    comp = ppe.utils.comparer.Comparer(outputs=["a"], concurrency=1)
    comp.add_engine("cpu", engine_cpu, *loaders_cpu)
    comp.add_engine("gpu", engine_gpu, *loaders_gpu)
    comp.compare()


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_compare_concurrency_wrong(engine_fn):
    loader = list(torch.ones(10) for _ in range(10))
    engine_cpu, loaders_cpu = engine_fn(Model, "cpu", [1.0], loader)
    engine_gpu, loaders_gpu = engine_fn(Model, "cuda:0", [0.5], loader)
    comp = ppe.utils.comparer.Comparer(outputs=["a"], concurrency=1)
    comp.add_engine("cpu", engine_cpu, *loaders_cpu)
    comp.add_engine("gpu", engine_gpu, *loaders_gpu)
    with pytest.raises(AssertionError):
        comp.compare()


class ModelForComparer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(10, 10, 3, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(10),
            torch.nn.Linear(3, 1),
        )

    def forward(self, x):
        # The return value depends only on the argument.
        return x.sum()


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_model_comparer(engine_fn):
    loader = list(torch.ones(2, 10, 10, 10) for _ in range(10))
    engine_cpu, loaders_cpu = engine_fn(ModelForComparer, "cpu", [], loader)
    engine_gpu, loaders_gpu = engine_fn(ModelForComparer, "cuda:0", [], loader)
    comp = ppe.utils.comparer.Comparer(outputs=["a"])
    compare_fn = ppe.utils.comparer.get_default_comparer(rtol=1e-2, atol=1e-2)
    comp = ppe.utils.comparer.Comparer(compare_fn=compare_fn, params=True)
    comp.add_engine("cpu", engine_cpu, *loaders_cpu)
    comp.add_engine("gpu", engine_gpu, *loaders_gpu)
    comp.compare()


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_model_comparer_invalid(engine_fn):
    loader = list(torch.ones(2, 10, 10, 10) for _ in range(10))
    engine_cpu, loaders_cpu = engine_fn(ModelForComparer, "cpu", [], loader, seed=0)
    engine_gpu, loaders_gpu = engine_fn(ModelForComparer, "cuda:0", [], loader, seed=1)
    comp = ppe.utils.comparer.Comparer(outputs=["a"])
    compare_fn = ppe.utils.comparer.get_default_comparer(rtol=1e-2, atol=1e-2)
    comp = ppe.utils.comparer.Comparer(compare_fn=compare_fn, params=True)
    comp.add_engine("cpu", engine_cpu, *loaders_cpu)
    comp.add_engine("gpu", engine_gpu, *loaders_gpu)
    with pytest.raises(AssertionError):
        comp.compare()


class ModelRetTuple(torch.nn.Module):
    def __init__(self, device, ret_val):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(10))
        self.device = device
        self.ret_val = ret_val

    def forward(self, x):
        a = torch.tensor(self.ret_val, device=self.device)
        a.requires_grad = True
        return (a, x)


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_compare_tuple_output(engine_fn):
    loader = list(torch.ones(10) for _ in range(10))
    engine_cpu, loaders_cpu = engine_fn(ModelRetTuple, "cpu", [1.0], loader)
    engine_gpu, loaders_gpu = engine_fn(ModelRetTuple, "cuda:0", [1.0], loader)
    comp = ppe.utils.comparer.Comparer()
    comp.add_engine("cpu", engine_cpu, *loaders_cpu)
    comp.add_engine("gpu", engine_gpu, *loaders_gpu)
    comp.compare()


class Output(typing.NamedTuple):
    a: torch.Tensor
    x: torch.Tensor


class ModelRetNamedTuple(torch.nn.Module):
    def __init__(self, device, ret_val):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(10))
        self.device = device
        self.ret_val = ret_val

    def forward(self, x):
        a = torch.tensor(self.ret_val, device=self.device)
        a.requires_grad = True
        return Output(a, x)


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_compare_namedtuple_output(engine_fn):
    loader = list(torch.ones(10) for _ in range(10))
    engine_cpu, loaders_cpu = engine_fn(ModelRetNamedTuple, "cpu", [1.0], loader)
    engine_gpu, loaders_gpu = engine_fn(ModelRetNamedTuple, "cuda:0", [1.0], loader)
    comp = ppe.utils.comparer.Comparer()
    comp.add_engine("cpu", engine_cpu, *loaders_cpu)
    comp.add_engine("gpu", engine_gpu, *loaders_gpu)
    comp.compare()


class MyModel(torch.nn.Module):
    def __init__(self, device, offset):
        super().__init__()
        self.model = torch.nn.Linear(20, 10)
        self.offset = offset

    def forward(self, x, t):
        y = self.model(x)
        prefix = 'train' if self.training else 'val'
        loss = F.l1_loss(y, t)
        ppe.reporting.report({prefix + '/loss': loss})
        return loss + self.offset


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_jit_runtime_output_comparer(engine_fn):
    loader = torch.utils.data.DataLoader(
        [(torch.rand(20,), torch.rand(10,)) for i in range(100)])
    ppe.runtime.runtime_registry.register("jit-cpu", JITRuntime)
    engine_cpu, loaders_cpu = engine_fn(MyModel, "cpu", [0.0], loader)
    engine_gpu, loaders_gpu = engine_fn(MyModel, "cuda:0", [0.0], loader)
    engine_jit, loaders_jit = engine_fn(MyModel, "jit-cpu", [0.0], loader)
    comp = ppe.utils.comparer.Comparer()
    comp.add_engine('cpu', engine_cpu, *loaders_cpu)
    comp.add_engine('gpu', engine_gpu, *loaders_gpu)
    comp.add_engine('jit', engine_jit, *loaders_jit)
    comp.compare()


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_jit_runtime_output_comparer_invalid(engine_fn):
    loader = torch.utils.data.DataLoader(
        [(torch.rand(20,), torch.rand(10,)) for i in range(100)])
    ppe.runtime.runtime_registry.register("jit-cpu", JITRuntime)
    engine_cpu, loaders_cpu = engine_fn(MyModel, "cpu", [0.0], loader)
    engine_gpu, loaders_gpu = engine_fn(MyModel, "cuda:0", [0.0], loader)
    engine_jit, loaders_jit = engine_fn(MyModel, "jit-cpu", [0.5], loader)
    comp = ppe.utils.comparer.Comparer()
    comp.add_engine('cpu', engine_cpu, *loaders_cpu)
    comp.add_engine('gpu', engine_gpu, *loaders_gpu)
    comp.add_engine('jit', engine_jit, *loaders_jit)
    with pytest.raises(AssertionError):
        comp.compare()


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_jit_runtime_model_comparer(engine_fn):
    loader = torch.utils.data.DataLoader([torch.rand(20,) for i in range(100)])
    ppe.runtime.runtime_registry.register("jit-cpu", JITRuntime)
    engine_cpu, loaders_cpu = engine_fn(ModelForComparer, "cpu", [], loader)
    engine_gpu, loaders_gpu = engine_fn(ModelForComparer, "cuda:0", [], loader)
    engine_jit, loaders_jit = engine_fn(ModelForComparer, "jit-cpu", [], loader)
    comp = ppe.utils.comparer.Comparer(params=True)
    comp.add_engine('cpu', engine_cpu, *loaders_cpu)
    comp.add_engine('gpu', engine_gpu, *loaders_gpu)
    comp.add_engine('jit', engine_jit, *loaders_jit)
    comp.compare()


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_jit_runtime_model_comparer_invalid(engine_fn):
    loader = torch.utils.data.DataLoader([torch.rand(20,) for i in range(100)])
    ppe.runtime.runtime_registry.register("jit-cpu", JITRuntime)
    engine_cpu, loaders_cpu = engine_fn(ModelForComparer, "cpu", [], loader, seed=0)
    engine_gpu, loaders_gpu = engine_fn(ModelForComparer, "cuda:0", [], loader, seed=0)
    engine_jit, loaders_jit = engine_fn(ModelForComparer, "jit-cpu", [], loader, seed=1)
    comp = ppe.utils.comparer.Comparer(params=True)
    comp.add_engine('cpu', engine_cpu, *loaders_cpu)
    comp.add_engine('gpu', engine_gpu, *loaders_gpu)
    comp.add_engine('jit', engine_jit, *loaders_jit)
    with pytest.raises(AssertionError):
        comp.compare()


class ModelForIntermediateValue(torch.nn.Module):
    def __init__(self, device, intermediate_value):
        super().__init__()
        self.model = torch.nn.Linear(20, 10)
        self.hidden = torch.nn.Parameter(torch.full((10,), intermediate_value))

    def forward(self, x, t):
        y = self.model(x)
        for i in range(5):
            ppe.utils.comparer.intermediate_value('y', y + self.hidden + i)
        loss = F.l1_loss(y, t)
        return loss


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_compare_intermediate(engine_fn):
    loader = torch.utils.data.DataLoader(
        [(torch.rand(20,), torch.rand(10,)) for i in range(100)])
    ppe.runtime.runtime_registry.register("jit-cpu", JITRuntime)
    engine_cpu, loaders_cpu = engine_fn(
        ModelForIntermediateValue, "cpu", [10.0], loader)
    engine_gpu, loaders_gpu = engine_fn(
        ModelForIntermediateValue, "cuda:0", [10.0], loader)
    engine_jit, loaders_jit = engine_fn(
        ModelForIntermediateValue, "jit-cpu", [10.0], loader)
    comp = ppe.utils.comparer.Comparer()
    comp.add_engine('cpu', engine_cpu, *loaders_cpu)
    comp.add_engine('gpu', engine_gpu, *loaders_gpu)
    comp.add_engine('jit', engine_jit, *loaders_jit)
    comp.compare()


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_compare_intermediate_invalid(engine_fn):
    loader = torch.utils.data.DataLoader(
        [(torch.rand(20,), torch.rand(10,)) for i in range(100)])
    ppe.runtime.runtime_registry.register("jit-cpu", JITRuntime)
    engine_cpu, loaders_cpu = engine_fn(
        ModelForIntermediateValue, "cpu", [10.0], loader)
    engine_gpu, loaders_gpu = engine_fn(
        ModelForIntermediateValue, "cuda:0", [10.0], loader)
    engine_jit, loaders_jit = engine_fn(
        ModelForIntermediateValue, "jit-cpu", [11.1], loader)
    comp = ppe.utils.comparer.Comparer()
    comp.add_engine('cpu', engine_cpu, *loaders_cpu)
    comp.add_engine('gpu', engine_gpu, *loaders_gpu)
    comp.add_engine('jit', engine_jit, *loaders_jit)
    with pytest.raises(AssertionError):
        comp.compare()


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_compare_async(engine_fn):
    def create_model(device):
        return test_trainer.MyModelWithLossAsync(test_trainer.MyModel())
    loader = torch.utils.data.DataLoader(
        [(torch.rand(20,), torch.rand(10,)) for i in range(10)])
    engine_cpu, loaders_cpu = engine_fn(create_model, "cpu", [], loader)
    engine_gpu, loaders_gpu = engine_fn(create_model, "cuda:0", [], loader)
    comp = ppe.utils.comparer.Comparer()
    comp.add_engine("cpu", engine_cpu, *loaders_cpu)
    comp.add_engine("gpu", engine_gpu, *loaders_gpu)
    comp.compare()


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
@pytest.mark.parametrize("model_class", [MyModel, ModelForIntermediateValue])
@pytest.mark.parametrize("params", [False, True])
def test_dump(engine_fn, model_class, params):
    loader = torch.utils.data.DataLoader(
        [(torch.rand(20,), torch.rand(10,)) for i in range(5)])
    ppe.runtime.runtime_registry.register("jit-cpu", JITRuntime)
    engine_cpu, loaders_cpu = engine_fn(model_class, "cpu", [1.0], loader)
    engine_gpu, loaders_gpu = engine_fn(model_class, "cuda:0", [1.0], loader)
    engine_jit, loaders_jit = engine_fn(model_class, "jit-cpu", [1.0], loader)
    comp = ppe.utils.comparer.Comparer(params=params)
    with tempfile.TemporaryDirectory() as tmpdir:
        comp.dump(engine_cpu, f'{tmpdir}/cpu', *loaders_cpu)
        comp.dump(engine_gpu, f'{tmpdir}/gpu', *loaders_gpu)
        comp.dump(engine_jit, f'{tmpdir}/jit', *loaders_jit)
        comp.add_dump('cpu', f'{tmpdir}/cpu')
        comp.add_dump('gpu', f'{tmpdir}/gpu')
        comp.add_dump('jit', f'{tmpdir}/jit')
        comp.compare()


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
@pytest.mark.parametrize("model_class", [
    MyModel, ModelForIntermediateValue])
def test_dump_invalid(engine_fn, model_class):
    loader = torch.utils.data.DataLoader(
        [(torch.rand(20,), torch.rand(10,)) for i in range(5)])
    ppe.runtime.runtime_registry.register("jit-cpu", JITRuntime)
    engine_cpu, loaders_cpu = engine_fn(model_class, "cpu", [1.0], loader)
    engine_gpu, loaders_gpu = engine_fn(model_class, "cuda:0", [1.0], loader)
    engine_jit, loaders_jit = engine_fn(model_class, "jit-cpu", [2.0], loader)
    comp = ppe.utils.comparer.Comparer()
    with tempfile.TemporaryDirectory() as tmpdir1, \
         tempfile.TemporaryDirectory() as tmpdir2, \
         tempfile.TemporaryDirectory() as tmpdir3:
        comp.dump(engine_cpu, tmpdir1, *loaders_cpu)
        comp.dump(engine_gpu, tmpdir2, *loaders_gpu)
        comp.dump(engine_jit, tmpdir3, *loaders_jit)
        comp.add_dump('cpu', tmpdir1)
        comp.add_dump('gpu', tmpdir2)
        comp.add_dump('jit', tmpdir3)
        with pytest.raises(AssertionError):
            comp.compare()


@pytest.mark.gpu
@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_compare_baseline(engine_fn):
    loader = list(torch.ones(10) for _ in range(10))
    engine_cpu, loaders_cpu = engine_fn(Model, "cpu", [1.0], loader)
    engine_gpu, loaders_gpu = engine_fn(Model, "cuda:0", [1.0], loader)
    comp = ppe.utils.comparer.Comparer(baseline="cpu")
    comp.add_engine("cpu", engine_cpu, *loaders_cpu)
    comp.add_engine("gpu", engine_gpu, *loaders_gpu)
    comp.compare()
