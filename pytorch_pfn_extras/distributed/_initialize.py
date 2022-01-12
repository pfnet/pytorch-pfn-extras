import os
from typing import Tuple

import torch


def initialize_ompi_environment(
    *,
    backend: str = "gloo",
    init_method: str = "tcp",
    world_size: int = 1,
    rank: int = 0,
    local_rank: int = 0,
    addr: str = "localhost",
    port: str = "1234"
) -> Tuple[int, int, int]:
    """Initialize `torch.distributed` environments with values taken from
    OpenMPI.

    Args:
        backend: The backend to be used, only ``"gloo"`` and ``"nccl"`` are
            supported.  Defaults to ``"gloo"``.
        init_method: Initialization method used by torch, only ``"tcp"`` and
            ``"env"`` are supported. Defaults to ``"tcp"``.
        world_size: The total world size to be used in case it is not specified
            in MPI env vars. Defaults to ``1``.
        rank: The process rank to be used in case it is not specified in MPI
            env vars. Defaults to ``0``.
        local_rank: The process local rank to be used in case it is not
            specified in MPI env vars. Defaults to ``0``.
        addr: The address of the master process of `torch.distributed`.
            Defaults to ``"localhost"``
        port: The port of the master process of `torch.distributed`.
            Defaults to ``"1234"``
    """
    e = os.environ
    backend = backend
    # Ranks determined from mpirun
    world_size = int(e.get("OMPI_COMM_WORLD_SIZE", world_size))
    rank = int(e.get("OMPI_COMM_WORLD_RANK", rank))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", local_rank))
    addr = e.get("MASTER_ADDR", addr)
    port = e.get("MASTER_PORT", port)

    if backend not in ("gloo" ,"nccl"):
        raise ValueError(
            "Invalid value for backend, only 'gloo' and 'nccl' are supported")
    if init_method == "env":
        init_method = "env://"
        e["MASTER_ADDR"] = addr
        e["MASTER_PORT"] = port
        e["WORLD_SIZE"] = str(world_size)
        e["RANK"] = str(rank)
        e["LOCAL_RANK"] = str(local_rank)
    elif init_method == "tcp":
        init_method = f"tcp://{addr}:{port}"
    else:
        raise ValueError(
            "Invalid value for init_method, only 'env' and 'tcp' are supported")

    if world_size > 1 and not torch.distributed.is_initialized():  # type: ignore
        torch.distributed.init_process_group(  # type: ignore
            backend, init_method=init_method,
            world_size=world_size, rank=rank
        )
        torch.distributed.barrier()  # type: ignore

    return world_size, rank, local_rank
