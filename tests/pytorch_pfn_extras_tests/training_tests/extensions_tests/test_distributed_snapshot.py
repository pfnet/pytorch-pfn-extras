import glob
import os
import tempfile

import torch
import pytest

from pytorch_pfn_extras import distributed
from pytorch_pfn_extras import training
from pytorch_pfn_extras.training import extensions


def _create_distributed_model(gpu=True):
    comm_size, comm_rank, comm_local_rank, device = _init_distributed(True)

    device = torch.device('cuda:{}'.format(comm_local_rank)
                          if gpu else 'cpu')

    model = torch.nn.Linear(128, 1)
    if torch.distributed.is_initialized():
        if not gpu:
            raise pytest.skip("Distributed tests require GPUs.")
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device), device_ids=[comm_local_rank])
    else:
        model = model.to(device)

    return model


def get_trainer(path):
    epochs = 10  # FIXME
    model = _create_distributed_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizers = {'main': optimizer}
    models = {'main': model}
    return training.ExtensionsManager(
        models, optimizers, epochs, iters_per_epoch=1, out_dir=path)


def _init_distributed(use_cuda):
    if ('OMPI_COMM_WORLD_SIZE' in os.environ):
        size, rank, local_rank = (
            distributed.initialize_ompi_environment(
                backend="nccl", init_method="env"))
    else:
        pytest.skip("This test requires MPI to run")

    device = torch.device(
        "cuda:{}".format(local_rank) if use_cuda else "cpu")

    return size, rank, local_rank, device


@pytest.fixture(scope='function')
def path():
    with tempfile.TemporaryDirectory() as t_path:
        yield t_path


def _model_params_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0.0:
            return False
    return True


@pytest.mark.gpu
def test_distributed_snapshot(path):
    comm_size, comm_rank, comm_local_rank, device = _init_distributed(False)

    if comm_size > 1:
        torch.distributed.barrier()

    saver_rank = 0
    fmt = 'snapshot_iter_{.iteration}'
    snapshot = extensions.snapshot(filename=fmt, saver_rank=saver_rank)

    trainer = get_trainer(path)
    trainer.extend(snapshot, trigger=(1, 'iteration'), priority=2)
    for _ in range(1):
        with trainer.run_iteration():
            pass
    assert 1 == trainer.iteration
    pattern = os.path.join(trainer.out, "snapshot_iter_*")
    found = [os.path.basename(path) for path in glob.glob(pattern)]
    # the snapshot is generated only for the saver rank
    assert comm_rank == saver_rank and len(found) == 1 or len(found) == 0

    if comm_rank == saver_rank:
        new_trainer = get_trainer(path)
        new_trainer.load_state_dict(torch.load(os.path.join(path, found[0])))
        assert _model_params_equal(
            trainer._models['main'], new_trainer._models['main'])

    if comm_size > 1:
        torch.distributed.barrier()
