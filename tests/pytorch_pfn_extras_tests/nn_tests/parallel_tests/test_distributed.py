import os
import sys
import urllib.request
import tempfile

import numpy as np
import pytest
import pytorch_pfn_extras
import torch
from torch import multiprocessing as mp
from torch import distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

from pytorch_pfn_extras.nn.parallel import DistributedDataParallel


context = mp.get_context("spawn")


class Hooks:
    @staticmethod
    def _to_zero_hook(module):
        for param in module.parameters():
            param.grad = torch.zeros_like(param.data)


class Steps:
    @staticmethod
    def _step(module, input):
        output = module(input)
        output.backward()
        return output

    @staticmethod
    def _step_with_no_sync(module, input):
        with module.no_sync():
            output = module(input)
            output.backward()
        return output

    @staticmethod
    def _step_with_hook(module, input):
        module.register_comm_hook(Hooks._to_zero_hook)
        output = module(input)
        output.backward()
        return output


class Collectives:
    @staticmethod
    def _to_zero(values, group):
        for value in values:
            value.zero_()


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.param0 = nn.Parameter(torch.tensor(-1.))
        self.param1 = nn.Parameter(torch.tensor(1.))
        buf = torch.zeros(1)
        self.register_buffer("buffer", buf)

    def forward(self, x):
        x = x.sum()
        self.buffer = self.buffer + x
        if x > 0:
            return self.param0 * torch.abs(x)
        else:
            return self.param1 * torch.abs(x)


class MyModuleWithCheckpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(1, 1)
        self.l1 = nn.Linear(1, 1)
        self.l2 = nn.Linear(1, 1)

    def forward(self, x):
        y = self.l0(x)
        y = checkpoint(self.l1, y)
        y = checkpoint(self.l2, y)
        return y


def _run(init_file, input, module, rank, args, step, device_type):
    init_method = "file://{}".format(urllib.request.pathname2url(init_file))
    dist.init_process_group(backend="gloo",
                            init_method=init_method,
                            world_size=2, rank=rank)
    if device_type == "cpu":
        device = torch.device(device_type)
    elif device_type == "cuda":
        device = torch.device("{}:{}".format(device_type, rank))
        torch.cuda.set_device(device)
    else:
        raise AssertionError
    module.to(device)
    module = DistributedDataParallel(module, **args)

    output = step(module, input)

    grads = {n: p.grad for n, p in module.named_parameters()}
    return output.detach(), module.state_dict(), grads


def _launch(inputs,
            modules=None,
            args=None,
            step=Steps._step,
            device_type="cpu"):
    procs = []
    with tempfile.TemporaryDirectory() as tmpdir, context.Pool(2) as pool:
        if modules is None:
            modules = [MyModule(), MyModule()]
        if args is None:
            args = {}

        file = os.path.join(tmpdir, "init")
        for i, (input, module) in enumerate(zip(inputs, modules)):
            p = pool.apply_async(
                _run,
                args=(file, input, module, i, args, step,
                      device_type))
            procs.append(p)
        return [p.get() for p in procs]


def _device_types():
    retval = ["cpu"]
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        retval.append("cuda")
    return retval


@pytest.mark.skipif(
    sys.platform == 'win32',
    reason='DDP not fully supported on Windows')
class TestDistributedDataParallel:
    def test_save_load(self):
        module = MyModule()
        with_ddp = DistributedDataParallel(module)
        assert module.state_dict().keys() == with_ddp.state_dict().keys()
        module.load_state_dict(with_ddp.state_dict())
        assert np.array_equal(module.state_dict()["param0"],
                              with_ddp.state_dict()["param0"])
        assert np.array_equal(module.state_dict()["param1"],
                              with_ddp.state_dict()["param1"])
        assert np.array_equal(module.state_dict()["buffer"],
                              with_ddp.state_dict()["buffer"])

    @pytest.mark.parametrize('device_type', _device_types())
    def test_sync_init_params(self, device_type):
        module0 = MyModule()
        module0.param0.data = torch.tensor([1.])
        r0, r1 = _launch(
            inputs=[torch.tensor([1.]), torch.tensor([2.])],
            modules=[module0, MyModule()],
            device_type=device_type)
        assert r0[0].item() == 1
        assert r1[0].item() == 2
        assert r0[1]["param0"].item() == 1.0
        assert r1[1]["param0"].item() == 1.0

    @pytest.mark.parametrize('device_type', _device_types())
    def test_all_reduce(self, device_type):
        r0, r1 = _launch(
            inputs=[torch.tensor([1.]), torch.tensor([2.])],
            device_type=device_type)
        assert r0[0].item() == -1
        assert r1[0].item() == -2
        assert r0[2]["module.param0"].item() == 1.5
        assert r1[2]["module.param0"].item() == 1.5
        assert r0[2]["module.param1"] is None
        assert r1[2]["module.param1"] is None

    @pytest.mark.parametrize('device_type', _device_types())
    def test_specific_reduce(self, device_type):
        r0, r1 = _launch(
            inputs=[torch.tensor([1.]), torch.tensor([2.])],
            args={"reduce_function": Collectives._to_zero},
            device_type=device_type)
        assert r0[2]["module.param0"].item() == 0.0
        assert r1[2]["module.param0"].item() == 0.0

    @pytest.mark.parametrize('device_type', _device_types())
    def test_nosync_buffer(self, device_type):
        r0, r1 = _launch(
            inputs=[torch.tensor([1.]), torch.tensor([2.])],
            args={"broadcast_buffers": False},
            device_type=device_type)
        assert r0[0].item() == -1
        assert r1[0].item() == -2
        assert r0[1]["buffer"].item() == 1
        assert r1[1]["buffer"].item() == 2

    @pytest.mark.parametrize('device_type', _device_types())
    def test_sync_buffer(self, device_type):
        r0, r1 = _launch(
            inputs=[torch.tensor([1.]), torch.tensor([2.])],
            args={"broadcast_buffers": True},
            device_type=device_type)
        assert r0[0].item() == -1
        assert r1[0].item() == -2
        assert r0[1]["buffer"].item() == 1
        assert r1[1]["buffer"].item() == 1

    @pytest.mark.parametrize('device_type', _device_types())
    def test_specific_broadcast(self, device_type):
        r0, r1 = _launch(
            inputs=[torch.tensor([1.]), torch.tensor([2.])],
            args={"broadcast_function": Collectives._to_zero,
                  "broadcast_buffers": True},
            device_type=device_type)
        assert r0[1]["buffer"].item() == 0.0
        assert r1[1]["buffer"].item() == 0.0

    @pytest.mark.parametrize('device_type', _device_types())
    def test_define_by_run(self, device_type):
        r0, r1 = _launch(
            inputs=[torch.tensor([1.]), torch.tensor([-1])],
            device_type=device_type)
        assert r0[0].item() == -1
        assert r1[0].item() == 1
        assert r0[2]["module.param0"].item() == 0.5
        assert r1[2]["module.param0"].item() == 0.5
        assert r0[2]["module.param1"].item() == 0.5
        assert r1[2]["module.param1"].item() == 0.5

    @pytest.mark.parametrize('device_type', _device_types())
    def test_no_sync(self, device_type):
        r0, r1 = _launch(
            inputs=[torch.tensor([1.]), torch.tensor([2.])],
            step=Steps._step_with_no_sync,
            device_type=device_type)
        assert r0[0].item() == -1
        assert r1[0].item() == -2
        assert r0[2]["module.param0"].item() == 1
        assert r1[2]["module.param0"].item() == 2
        assert r0[2]["module.param1"] is None
        assert r1[2]["module.param1"] is None

    @pytest.mark.parametrize('device_type', _device_types())
    def test_hook(self, device_type):
        r0, r1 = _launch(
            inputs=[torch.tensor([1.]), torch.tensor([2.])],
            step=Steps._step_with_hook,
            device_type=device_type)
        assert r0[0].item() == -1
        assert r1[0].item() == -2
        assert r0[2]["module.param0"].item() == 0
        assert r1[2]["module.param0"].item() == 0
        assert r0[2]["module.param1"].item() == 0
        assert r1[2]["module.param1"].item() == 0

    @pytest.mark.parametrize('device_type', _device_types())
    @pytest.mark.skipif(
        not pytorch_pfn_extras.requires("1.6.0"),
        reason="Variable._execution_engine.queue_callback does not work "
               "with checkpointing when torch < 1.6.0")
    def test_checkpoint(self, device_type):
        r0, r1 = _launch(
            inputs=[torch.tensor([[1.]]), torch.tensor([[2.]])],
            modules=[MyModuleWithCheckpoint(), MyModuleWithCheckpoint()],
            step=Steps._step_with_hook,
            device_type=device_type)
        grad0 = r0[2]
        grad1 = r1[2]
        for key in grad0.keys():
            assert np.array_equal(grad0[key].cpu().numpy(),
                                  grad1[key].cpu().numpy())
