import typing

import pytest
import torch

import pytorch_pfn_extras as ppe


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


def _get_trainer(device, ret_val, model_class=Model):
    model = model_class(device, ret_val)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    trainer = ppe.engine.create_trainer(model, optimizer, 1, device=device)
    return trainer


def _get_evaluator(device, ret_val, model_class=Model):
    model = model_class(device, ret_val)
    evaluator = ppe.engine.create_evaluator(model, device=device)
    return evaluator


def _get_trainer_with_evaluator(device, ret_val, model_class=Model):
    model = model_class(device, ret_val)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    evaluator = ppe.engine.create_evaluator(model, device=device)
    trainer = ppe.engine.create_trainer(
        model, optimizer, 1, device=device, evaluator=evaluator)
    return trainer


@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_compare_every_iter(engine_fn):
    engine_cpu = engine_fn("cpu", 1.0)
    engine_gpu = engine_fn("cuda:0", 1.0)
    comp = ppe.utils.comparer.OutputsComparer(
        {"cpu": engine_cpu, "gpu": engine_gpu}, "a"
    )
    train_1 = list(torch.ones(10) for _ in range(10))
    train_2 = list(torch.ones(10) for _ in range(10))
    if engine_fn is _get_trainer_with_evaluator:
        eval_1 = list(torch.ones(10) for _ in range(10))
        eval_2 = list(torch.ones(10) for _ in range(10))
        comp.compare({"cpu": (train_1, eval_1), "gpu": (train_2, eval_2)})
    else:
        comp.compare({"cpu": train_1, "gpu": train_2})


@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_comparer_wrong(engine_fn):
    engine_cpu = engine_fn("cpu", 1.0)
    engine_gpu = engine_fn("cuda:0", 0.5)
    comp = ppe.utils.comparer.OutputsComparer(
        {"cpu": engine_cpu, "gpu": engine_gpu}, "a"
    )
    train_1 = list(torch.ones(10) for _ in range(10))
    train_2 = list(torch.ones(10) for _ in range(10))
    with pytest.raises(AssertionError):
        if engine_fn is _get_trainer_with_evaluator:
            eval_1 = list(torch.ones(10) for _ in range(10))
            eval_2 = list(torch.ones(10) for _ in range(10))
            comp.compare({"cpu": (train_1, eval_1), "gpu": (train_2, eval_2)})
        else:
            comp.compare({"cpu": train_1, "gpu": train_2})


class _CustomComparer:
    def __init__(self, n_iters=None):
        self.times_called = 0
        self.n_iters = n_iters

    def __call__(self, eng_name_1, eng_name_2, out_name, out_1, out_2):
        assert out_name in ("iter")
        assert eng_name_1 in ("cpu", "gpu")
        assert eng_name_1 != eng_name_2
        assert out_1 == out_2
        if out_name == "iter":
            self.times_called += 1
            assert out_1 == self.times_called * self.n_iters


@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_comparer_n_iters(engine_fn):
    engine_cpu = engine_fn("cpu", 1.0)
    engine_gpu = engine_fn("cuda:0", 1.0)
    n_iters = 3
    comp = ppe.utils.comparer.OutputsComparer(
        {"cpu": engine_cpu, "gpu": engine_gpu},
        ("iter",),
        compare_fn=_CustomComparer(n_iters),
    )
    train_1 = list(torch.ones(10) for _ in range(10))
    train_2 = list(torch.ones(10) for _ in range(10))
    if engine_fn is _get_trainer_with_evaluator:
        eval_1 = list(torch.ones(10) for _ in range(10))
        eval_2 = list(torch.ones(10) for _ in range(10))
        comp.compare(
            {"cpu": (train_1, eval_1), "gpu": (train_2, eval_2)},
            n_iters=n_iters)
        assert comp.compare_fn.times_called == 6
    else:
        comp.compare({"cpu": train_1, "gpu": train_2}, n_iters=n_iters)
        assert comp.compare_fn.times_called == 3


@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_comparer_kwargs(engine_fn):
    engine_cpu = engine_fn("cpu", 1.0)
    engine_gpu = engine_fn("cuda:0", 0.991)
    compare_fn = ppe.utils.comparer.get_default_comparer(rtol=1e-2, atol=1e-2)
    comp = ppe.utils.comparer.OutputsComparer(
        {"cpu": engine_cpu, "gpu": engine_gpu}, "a",
        compare_fn=compare_fn,
    )
    train_1 = list(torch.ones(10) for _ in range(10))
    train_2 = list(torch.ones(10) for _ in range(10))
    if engine_fn is _get_trainer_with_evaluator:
        eval_1 = list(torch.ones(10) for _ in range(10))
        eval_2 = list(torch.ones(10) for _ in range(10))
        comp.compare({"cpu": (train_1, eval_1), "gpu": (train_2, eval_2)})
    else:
        comp.compare({"cpu": train_1, "gpu": train_2})


@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_comparer_incompat_trigger(engine_fn):
    model_cpu = Model("cpu", 1.0)
    optimizer_cpu = torch.optim.SGD(model_cpu.parameters(), lr=1.0)
    trainer_cpu = ppe.engine.create_trainer(
        model_cpu, optimizer_cpu, 1, device="cpu",
    )

    model_gpu = Model("cuda:0", 1.0)
    optimizer_gpu = torch.optim.SGD(model_gpu.parameters(), lr=1.0)
    trainer_gpu = ppe.engine.create_trainer(
        model_gpu, optimizer_gpu, 1, device="cuda:0",
        stop_trigger=(1, "iteration"),
    )

    comp = ppe.utils.comparer.OutputsComparer(
        {"cpu": trainer_cpu, "gpu": trainer_gpu}, "a",
    )
    train_1 = list(torch.ones(10) for _ in range(10))
    train_2 = list(torch.ones(10) for _ in range(10))
    with pytest.raises(ValueError):
        if engine_fn is _get_trainer_with_evaluator:
            eval_1 = list(torch.ones(10) for _ in range(10))
            eval_2 = list(torch.ones(10) for _ in range(10))
            comp.compare({"cpu": (train_1, eval_1), "gpu": (train_2, eval_2)})
        else:
            comp.compare({"cpu": train_1, "gpu": train_2})


@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_compare_concurrency(engine_fn):
    engine_cpu = engine_fn("cpu", 1.0)
    engine_gpu = engine_fn("cuda:0", 1.0)
    comp = ppe.utils.comparer.OutputsComparer(
        {"cpu": engine_cpu, "gpu": engine_gpu}, "a",
        concurrency=1,
    )
    train_1 = list(torch.ones(10) for _ in range(10))
    train_2 = list(torch.ones(10) for _ in range(10))
    if engine_fn is _get_trainer_with_evaluator:
        eval_1 = list(torch.ones(10) for _ in range(10))
        eval_2 = list(torch.ones(10) for _ in range(10))
        comp.compare({"cpu": (train_1, eval_1), "gpu": (train_2, eval_2)})
    else:
        comp.compare({"cpu": train_1, "gpu": train_2})


@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_compare_concurrency_wrong(engine_fn):
    engine_cpu = engine_fn("cpu", 1.0)
    engine_gpu = engine_fn("cuda:0", 0.5)
    comp = ppe.utils.comparer.OutputsComparer(
        {"cpu": engine_cpu, "gpu": engine_gpu}, "a",
        concurrency=1,
    )
    train_1 = list(torch.ones(10) for _ in range(10))
    train_2 = list(torch.ones(10) for _ in range(10))
    with pytest.raises(AssertionError):
        if engine_fn is _get_trainer_with_evaluator:
            eval_1 = list(torch.ones(10) for _ in range(10))
            eval_2 = list(torch.ones(10) for _ in range(10))
            comp.compare({"cpu": (train_1, eval_1), "gpu": (train_2, eval_2)})
        else:
            comp.compare({"cpu": train_1, "gpu": train_2})


class ModelForComparer(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(10, 10, 3, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(10),
            torch.nn.Linear(3, 1),
        )

    def forward(self, x):
        return {"y": self.model(x).sum()}


def test_model_comparer():
    model_cpu = ModelForComparer()
    model_gpu = ModelForComparer()
    # Make the models to have the same initial weights
    model_gpu.load_state_dict(model_cpu.state_dict())
    ppe.to(model_gpu, device='cuda:0')

    optimizer_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.01)
    trainer_cpu = ppe.engine.create_trainer(
        model_cpu, optimizer_cpu, 1, device='cpu')
    optimizer_gpu = torch.optim.SGD(model_gpu.parameters(), lr=0.01)
    trainer_gpu = ppe.engine.create_trainer(
        model_gpu, optimizer_gpu, 1, device='cuda:0')
    compare_fn = ppe.utils.comparer.get_default_comparer(rtol=1e-2, atol=1e-2)
    comp = ppe.utils.comparer.ModelComparer(
        {"cpu": trainer_cpu, "gpu": trainer_gpu},
        compare_fn=compare_fn
    )

    train_1 = list(torch.ones(2, 10, 10, 10) for _ in range(10))
    train_2 = list(torch.ones(2, 10, 10, 10) for _ in range(10))
    comp.compare({"cpu": train_1, "gpu": train_2})


def test_model_comparer_invalid():
    model_cpu = ModelForComparer()
    model_gpu = ModelForComparer()
    ppe.to(model_gpu, device='cuda:0')

    optimizer_cpu = torch.optim.SGD(model_cpu.parameters(), lr=0.01)
    trainer_cpu = ppe.engine.create_trainer(
        model_cpu, optimizer_cpu, 1, device='cpu')
    optimizer_gpu = torch.optim.SGD(model_gpu.parameters(), lr=0.01)
    trainer_gpu = ppe.engine.create_trainer(
        model_gpu, optimizer_gpu, 1, device='cuda:0')
    compare_fn = ppe.utils.comparer.get_default_comparer(rtol=1e-2, atol=1e-2)
    comp = ppe.utils.comparer.ModelComparer(
        {"cpu": trainer_cpu, "gpu": trainer_gpu},
        compare_fn=compare_fn
    )

    train_1 = list(torch.ones(2, 10, 10, 10) for _ in range(10))
    train_2 = list(torch.ones(2, 10, 10, 10) for _ in range(10))
    with pytest.raises(AssertionError):
        comp.compare({"cpu": train_1, "gpu": train_2})


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


@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_compare_tuple_output(engine_fn):
    engine_cpu = engine_fn("cpu", 1.0, model_class=ModelRetTuple)
    engine_gpu = engine_fn("cuda:0", 1.0, model_class=ModelRetTuple)
    comp = ppe.utils.comparer.OutputsComparer({"cpu": engine_cpu, "gpu": engine_gpu})
    train_1 = list(torch.ones(10) for _ in range(10))
    train_2 = list(torch.ones(10) for _ in range(10))
    if engine_fn is _get_trainer_with_evaluator:
        eval_1 = list(torch.ones(10) for _ in range(10))
        eval_2 = list(torch.ones(10) for _ in range(10))
        comp.compare({"cpu": (train_1, eval_1), "gpu": (train_2, eval_2)})
    else:
        comp.compare({"cpu": train_1, "gpu": train_2})


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


@pytest.mark.parametrize("engine_fn", [
    _get_trainer, _get_evaluator, _get_trainer_with_evaluator])
def test_compare_namedtuple_output(engine_fn):
    engine_cpu = engine_fn("cpu", 1.0, model_class=ModelRetNamedTuple)
    engine_gpu = engine_fn("cuda:0", 1.0, model_class=ModelRetNamedTuple)
    comp = ppe.utils.comparer.OutputsComparer({"cpu": engine_cpu, "gpu": engine_gpu})
    train_1 = list(torch.ones(10) for _ in range(10))
    train_2 = list(torch.ones(10) for _ in range(10))
    if engine_fn is _get_trainer_with_evaluator:
        eval_1 = list(torch.ones(10) for _ in range(10))
        eval_2 = list(torch.ones(10) for _ in range(10))
        comp.compare({"cpu": (train_1, eval_1), "gpu": (train_2, eval_2)})
    else:
        comp.compare({"cpu": train_1, "gpu": train_2})
