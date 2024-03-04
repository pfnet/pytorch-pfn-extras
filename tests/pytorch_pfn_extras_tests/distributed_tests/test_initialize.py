import torch.distributed
from pytorch_pfn_extras.distributed import initialize_ompi_environment


def test_initialize_ompi_environment_with_single_process():
    assert not torch.distributed.is_initialized()
    world_size, rank, local_rank = initialize_ompi_environment(
        backend="gloo", init_method="tcp"
    )
    assert torch.distributed.is_initialized()
    torch.distributed.destroy_process_group()
    assert not torch.distributed.is_initialized()
