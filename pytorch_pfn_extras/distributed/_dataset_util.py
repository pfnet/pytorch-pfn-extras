from typing import List, Optional

import numpy as np
import torch


def _shared_random_seed() -> int:
    seed = torch.randint(0, 2 ** 31, size=())
    if torch.distributed.is_initialized():  # type: ignore
        if torch.distributed.get_backend() == "nccl":  # type: ignore
            seed = seed.cuda()
        torch.distributed.broadcast(seed, 0)  # type: ignore
    return int(seed)


def create_distributed_subset_indices(
    num_total_samples: int,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> List[int]:
    """Returns a indices of a dataset to be used for the current process.

    Args:
        num_total_samples: The size of the dataset.
        num_replicas: Number of processes participating in the training.
            By default, ``torch.distributed.get_world_size()`` is used.
        rank: Rank of the current process within `num_replicas`.
            By default, ``torch.distributed.get_rank()`` is used.
        shuffle: If ``True`` (default), shuffle the indices.
        seed: Random seed used to shuffle.
    """
    if num_replicas is None:
        num_replicas = torch.distributed.get_world_size()  # type: ignore
    if rank is None:
        rank = torch.distributed.get_rank()  # type: ignore

    indices = list(range(num_total_samples))
    if shuffle:
        if seed is None:
            seed = _shared_random_seed()
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    n_sub_samples = (num_total_samples + num_replicas - 1) // num_replicas
    b = num_total_samples * rank // num_replicas
    e = b + n_sub_samples
    return indices[b:e]
