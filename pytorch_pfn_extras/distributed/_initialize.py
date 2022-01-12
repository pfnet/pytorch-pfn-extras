import os

import torch


def initialize_ompi_environment(
    *,
    dist_backend: str = "gloo",
    world_size: int = 1,
    rank: int = 0,
    local_rank: int = 0,
    addr: str = "localhost",
    port:int = 1234
) -> None:
    e = os.environ
    dist_backend = dist_backend
    world_size = int(e.get("OMPI_COMM_WORLD_SIZE", world_size))
    rank = int(e.get("OMPI_COMM_WORLD_RANK", rank))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", local_rank))
    addr = e.get("MASTER_ADDR", addr)
    port = int(e.get("MASTER_PORT", port))

    if world_size > 1:
        torch.distributed.init_process_group(  # type: ignore
            dist_backend, init_method=f"tcp://{addr}:{port}",
            world_size=world_size, rank=rank
        )
        torch.distributed.barrier()  # type: ignore
