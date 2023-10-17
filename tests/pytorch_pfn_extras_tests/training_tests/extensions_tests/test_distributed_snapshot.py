import glob
import os
import tempfile

import py
import pytest
import torch
import torch.distributed
from pytorch_pfn_extras import distributed, training
from pytorch_pfn_extras.training import extensions


def _create_distributed_model(gpu=True):
    comm_size, comm_rank, comm_local_rank, device = _init_distributed(True)

    device = torch.device("cuda:{}".format(comm_local_rank) if gpu else "cpu")

    model = torch.nn.Linear(128, 1)
    if torch.distributed.is_initialized():
        if not gpu:
            raise pytest.skip("Distributed tests require GPUs.")
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device), device_ids=[comm_local_rank]
        )
    else:
        model = model.to(device)

    return model


def get_trainer(path):
    epochs = 10  # FIXME
    model = _create_distributed_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizers = {"main": optimizer}
    models = {"main": model}
    return training.ExtensionsManager(
        models, optimizers, epochs, iters_per_epoch=1, out_dir=path
    )


def _init_distributed(use_cuda):
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        size, rank, local_rank = distributed.initialize_ompi_environment(
            backend="nccl", init_method="env"
        )
    else:
        pytest.skip("This test requires MPI to run")

    device = torch.device("cuda:{}".format(local_rank) if use_cuda else "cpu")

    return size, rank, local_rank, device


@pytest.fixture(scope="function")
def path():
    with tempfile.TemporaryDirectory() as t_path:
        yield t_path


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


def _model_params_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0.0:
            return False
    return True


@pytest.mark.gpu
@pytest.mark.mpi
@pytest.mark.parametrize("saver_rank", [0, 1])
def test_distributed_snapshot(mpi_tmp_path, saver_rank):
    comm_size, comm_rank, comm_local_rank, device = _init_distributed(False)

    if comm_size > 1:
        torch.distributed.barrier()

    fmt = "snapshot_iter_{.iteration}"
    snapshot = extensions.snapshot(filename=fmt, saver_rank=saver_rank)

    trainer = get_trainer(mpi_tmp_path)
    trainer.extend(snapshot, trigger=(1, "iteration"), priority=2)
    for _ in range(1):
        with trainer.run_iteration():
            pass
    assert 1 == trainer.iteration
    pattern = os.path.join(trainer.out, "snapshot_iter_*")
    found = [os.path.basename(path) for path in glob.glob(pattern)]
    # the snapshot is generated only for the saver rank
    assert len(found) == 1

    new_trainer = get_trainer(mpi_tmp_path)
    assert not _model_params_equal(
        trainer._models["main"], new_trainer._models["main"]
    )
    new_trainer.load_state_dict(
        torch.load(os.path.join(mpi_tmp_path, found[0]))
    )
    assert _model_params_equal(
        trainer._models["main"], new_trainer._models["main"]
    )


@pytest.mark.gpu
@pytest.mark.mpi
@pytest.mark.parametrize("saver_rank", [0, 1])
def test_distributed_snapshot_autoload(mpi_tmp_path, saver_rank):
    comm_size, comm_rank, comm_local_rank, device = _init_distributed(False)

    if comm_size > 1:
        torch.distributed.barrier()
    fmt = "snapshot_iter_{.iteration}"
    snapshot = extensions.snapshot(
        filename=fmt, saver_rank=saver_rank, autoload=True
    )
    trainer = get_trainer(mpi_tmp_path)
    trainer.extend(snapshot, trigger=(1, "iteration"), priority=2)
    for _ in range(1):
        with trainer.run_iteration():
            pass
    assert 1 == trainer.iteration
    pattern = os.path.join(trainer.out, "snapshot_iter_*")
    found = [os.path.basename(path) for path in glob.glob(pattern)]
    assert len(found) == 1
    new_trainer = get_trainer(mpi_tmp_path)
    assert not _model_params_equal(
        trainer._models["main"], new_trainer._models["main"]
    )
    snapshot.initialize(new_trainer)
    assert _model_params_equal(
        trainer._models["main"], new_trainer._models["main"]
    )
