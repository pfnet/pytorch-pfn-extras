import sys
from typing import Tuple

import pytest
import pytorch_pfn_extras as ppe

if not ppe.requires("2.1.0") or sys.platform == "win32":
    pytest.skip(
        "sharded snapshot is tested only with pytorch>2.1 or later.",
        allow_module_level=True,
    )
import os
from glob import glob
from io import BytesIO

import py
import torch
import torch.distributed
import torch.distributed.fsdp as fsdp
from pytorch_pfn_extras import distributed, training
from pytorch_pfn_extras.training.extensions import SnapshotMode, snapshot


def _init_distributed(use_cuda):
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        size, rank, local_rank = distributed.initialize_ompi_environment(
            backend="nccl",
            init_method="env",
            timeout=15,
        )
    else:
        pytest.skip("This test requires MPI to run")

    torch.cuda.set_device("cuda:{}".format(local_rank) if use_cuda else "cpu")
    device = torch.device("cuda:{}".format(local_rank) if use_cuda else "cpu")

    return size, rank, local_rank, device


def _create_fsdp_model(device):
    model = torch.nn.Linear(2**13 + 1, 3)
    model = fsdp.FullyShardedDataParallel(
        model, device_id=device, sync_module_states=False
    )
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=1.0)
    optimizer.zero_grad()
    input_tensor = torch.rand((4, 2**13 + 1))
    output = model.forward(input_tensor).mean()
    output.backward()
    optimizer.step()
    return model, optimizer


def _assert_state_dict_is_eq(actuary_state_dict, expected_state_dict):
    with BytesIO() as actuary_io, BytesIO() as expected_io:
        torch.save(actuary_state_dict, actuary_io)
        torch.save(expected_state_dict, expected_io)
        actuary_io.seek(0)
        expected_io.seek(0)
        assert actuary_io.read() == expected_io.read()


@pytest.mark.mpi
@pytest.mark.gpu
@pytest.mark.parametrize(
    "expected_state_dict_type,actuary_state_dict_type",
    [
        (
            (
                fsdp.StateDictType.FULL_STATE_DICT,
                fsdp.FullStateDictConfig(rank0_only=False),
                fsdp.FullOptimStateDictConfig(rank0_only=False),
            ),
            (
                fsdp.StateDictType.FULL_STATE_DICT,
                fsdp.FullStateDictConfig(rank0_only=False),
                fsdp.FullOptimStateDictConfig(rank0_only=False),
            ),
        ),
        (
            (
                fsdp.StateDictType.FULL_STATE_DICT,
                fsdp.FullStateDictConfig(rank0_only=False),
                fsdp.FullOptimStateDictConfig(rank0_only=False),
            ),
            (
                fsdp.StateDictType.SHARDED_STATE_DICT,
                fsdp.ShardedStateDictConfig(
                    offload_to_cpu=True, use_dtensor=False
                ),
                fsdp.ShardedOptimStateDictConfig(
                    offload_to_cpu=True, use_dtensor=False
                ),
            ),
        ),
        (
            (
                fsdp.StateDictType.SHARDED_STATE_DICT,
                fsdp.ShardedStateDictConfig(
                    offload_to_cpu=True, use_dtensor=False
                ),
                fsdp.ShardedOptimStateDictConfig(
                    offload_to_cpu=True, use_dtensor=False
                ),
            ),
            (
                fsdp.StateDictType.FULL_STATE_DICT,
                fsdp.FullStateDictConfig(rank0_only=False),
                fsdp.FullOptimStateDictConfig(rank0_only=False),
            ),
        ),
        (
            (
                fsdp.StateDictType.SHARDED_STATE_DICT,
                fsdp.ShardedStateDictConfig(
                    offload_to_cpu=True, use_dtensor=False
                ),
                fsdp.ShardedOptimStateDictConfig(
                    offload_to_cpu=True, use_dtensor=False
                ),
            ),
            (
                fsdp.StateDictType.SHARDED_STATE_DICT,
                fsdp.ShardedStateDictConfig(
                    offload_to_cpu=True, use_dtensor=False
                ),
                fsdp.ShardedOptimStateDictConfig(
                    offload_to_cpu=True, use_dtensor=False
                ),
            ),
        ),
    ],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sharded_state_dict(
    expected_state_dict_type: Tuple[
        fsdp.StateDictType,
        fsdp.FullStateDictConfig,
        fsdp.FullOptimStateDictConfig,
    ],
    actuary_state_dict_type: Tuple[
        fsdp.StateDictType,
        fsdp.FullStateDictConfig,
        fsdp.FullOptimStateDictConfig,
    ],
):
    size, rank, local_rank, device = _init_distributed(True)

    # create actuary state dict
    model, optimizer = _create_fsdp_model(device)
    fsdp.FullyShardedDataParallel.set_state_dict_type(
        model,
        *actuary_state_dict_type,
    )
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    actuary_model, actuary_optimizer = _create_fsdp_model(device)
    fsdp.FullyShardedDataParallel.set_state_dict_type(
        actuary_model,
        *actuary_state_dict_type,
    )
    actuary_model.load_state_dict(model_state_dict)
    actuary_optimizer.load_state_dict(optimizer_state_dict)

    # create expected state dict
    fsdp.FullyShardedDataParallel.set_state_dict_type(
        model, *expected_state_dict_type
    )
    model_state_dict = model.state_dict()
    optimizer_state_dict = fsdp.FullyShardedDataParallel.optim_state_dict(
        model,
        optimizer,
        optim_state_dict=optimizer.state_dict(),
    )

    expected_model, expected_optimizer = _create_fsdp_model(device)
    fsdp.FullyShardedDataParallel.set_state_dict_type(
        expected_model, *expected_state_dict_type
    )
    expected_model.load_state_dict(model_state_dict)
    expected_optimizer.load_state_dict(
        fsdp.FullyShardedDataParallel.optim_state_dict_to_load(
            expected_model, expected_optimizer, optimizer_state_dict
        )
    )

    # Full state dict check
    check_state_dict_type = (
        fsdp.StateDictType.FULL_STATE_DICT,
        fsdp.FullStateDictConfig(rank0_only=False),
        fsdp.FullOptimStateDictConfig(rank0_only=False),
    )
    fsdp.FullyShardedDataParallel.set_state_dict_type(
        actuary_model, *check_state_dict_type
    )
    actuary_model_state_dict = actuary_model.state_dict()
    actuary_optimizer_state_dict = (
        fsdp.FullyShardedDataParallel.optim_state_dict(
            actuary_model,
            actuary_optimizer,
            optim_state_dict=actuary_optimizer.state_dict(),
        )
    )

    fsdp.FullyShardedDataParallel.set_state_dict_type(
        expected_model, *check_state_dict_type
    )
    expected_model_state_dict = expected_model.state_dict()
    expected_optimizer_state_dict = (
        fsdp.FullyShardedDataParallel.optim_state_dict(
            expected_model,
            expected_optimizer,
            optim_state_dict=expected_optimizer.state_dict(),
        )
    )

    _assert_state_dict_is_eq(
        actuary_model_state_dict, expected_model_state_dict
    )
    _assert_state_dict_is_eq(
        actuary_optimizer_state_dict, expected_optimizer_state_dict
    )

    # sharded state dict check
    check_state_dict_type = (
        fsdp.StateDictType.SHARDED_STATE_DICT,
        fsdp.ShardedStateDictConfig(offload_to_cpu=True, use_dtensor=False),
        fsdp.ShardedOptimStateDictConfig(
            offload_to_cpu=True, use_dtensor=False
        ),
    )
    fsdp.FullyShardedDataParallel.set_state_dict_type(
        actuary_model, *check_state_dict_type
    )
    actuary_model_state_dict = actuary_model.state_dict()
    actuary_optimizer_state_dict = (
        fsdp.FullyShardedDataParallel.optim_state_dict(
            actuary_model,
            actuary_optimizer,
            optim_state_dict=actuary_optimizer.state_dict(),
        )
    )

    fsdp.FullyShardedDataParallel.set_state_dict_type(
        expected_model, *check_state_dict_type
    )
    expected_model_state_dict = expected_model.state_dict()
    expected_optimizer_state_dict = (
        fsdp.FullyShardedDataParallel.optim_state_dict(
            expected_model,
            expected_optimizer,
            optim_state_dict=expected_optimizer.state_dict(),
        )
    )

    _assert_state_dict_is_eq(
        actuary_model_state_dict, expected_model_state_dict
    )
    _assert_state_dict_is_eq(
        actuary_optimizer_state_dict, expected_optimizer_state_dict
    )


@pytest.fixture(scope="function")
def mpi_tmp_path(tmpdir):
    try:
        from mpi4py import MPI
    except ImportError:
        pytest.fail("mpi4py needs to be installed to run this test")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    name = str(tmpdir) if rank == 0 else None
    name = comm.bcast(name, root=0)
    yield py.path.local(name)
    comm.barrier()


def get_trainer(path, device):
    epochs = 20  # FIXME
    model, optimizer = _create_fsdp_model(device=device)
    optimizers = {"main": optimizer}
    models = {"main": model}
    return training.ExtensionsManager(
        models, optimizers, epochs, iters_per_epoch=1, out_dir=path
    )


@pytest.mark.mpi
@pytest.mark.gpu
def test_sharded_snapshot(mpi_tmp_path):
    comm_size, comm_rank, comm_local_rank, device = _init_distributed(True)

    if comm_size > 1:
        torch.distributed.barrier()

    saver_rank = 0
    fmt = "snapshot_iter_{.iteration}"
    snapshot_extension = snapshot(
        filename=fmt, snapshot_mode=SnapshotMode.SHARDED, saver_rank=saver_rank
    )

    trainer = get_trainer(mpi_tmp_path, device)
    trainer.extend(snapshot_extension, trigger=(1, "iteration"), priority=2)
    for _ in range(1):
        with trainer.run_iteration():
            pass

    if comm_size > 1:
        torch.distributed.barrier()

    pattern = os.path.join(trainer.out, "snapshot_iter_*")
    found = [path for path in glob(pattern)]

    assert len(found) == 1
    (snapshot_path,) = found
    complete_path = os.path.join(snapshot_path, "complete")
    assert os.path.exists(complete_path)


@pytest.mark.mpi
@pytest.mark.gpu
def test_sharded_snapshot_cleanup(mpi_tmp_path):
    comm_size, comm_rank, comm_local_rank, device = _init_distributed(True)

    if comm_size > 1:
        torch.distributed.barrier()

    saver_rank = 0
    fmt = "snapshot_iter_{.iteration}"
    snapshot_extension = snapshot(
        filename=fmt,
        snapshot_mode=SnapshotMode.SHARDED,
        saver_rank=saver_rank,
        n_retains=3,
    )

    trainer = get_trainer(mpi_tmp_path, device)
    trainer.extend(snapshot_extension, trigger=(1, "iteration"), priority=2)
    for _ in range(5):
        with trainer.run_iteration():
            pass
    if comm_size > 1:
        torch.distributed.barrier()

    pattern = os.path.join(trainer.out, "snapshot_iter_*")
    found = [os.path.basename(path) for path in glob(pattern)]

    assert len(found) <= 3


@pytest.mark.mpi
@pytest.mark.gpu
def test_sharded_snapshot_autoload(mpi_tmp_path):
    comm_size, comm_rank, comm_local_rank, device = _init_distributed(True)

    if comm_size > 1:
        torch.distributed.barrier()

    saver_rank = 0
    fmt = "snapshot_iter_{.iteration}"
    snapshot_extension = snapshot(
        filename=fmt,
        snapshot_mode=SnapshotMode.SHARDED,
        saver_rank=saver_rank,
        autoload=True,
    )

    trainer = get_trainer(mpi_tmp_path, device)
    trainer.extend(snapshot_extension, trigger=(1, "iteration"), priority=2)
    for _ in range(5):
        with trainer.run_iteration():
            pass

    if comm_size > 1:
        torch.distributed.barrier()

    trainer2 = get_trainer(mpi_tmp_path, device)
    snapshot_extension2 = snapshot(
        filename=fmt,
        snapshot_mode=SnapshotMode.SHARDED,
        saver_rank=saver_rank,
        autoload=False,
    )
    trainer2.extend(snapshot_extension2, trigger=(1, "iteration"), priority=2)

    with pytest.raises(AssertionError):
        _assert_state_dict_is_eq(trainer2.state_dict(), trainer.state_dict())

    trainer3 = get_trainer(mpi_tmp_path, device)
    snapshot_extension3 = snapshot(
        filename=fmt,
        snapshot_mode=SnapshotMode.SHARDED,
        saver_rank=saver_rank,
        autoload=True,
    )
    trainer3.extend(snapshot_extension3, trigger=(1, "iteration"), priority=2)

    _assert_state_dict_is_eq(trainer3.state_dict(), trainer.state_dict())
