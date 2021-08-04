from typing import Generic, Iterator, Optional, Sequence, TypeVar

import numpy as np
import torch
from torch.utils.data import Sampler, SubsetRandomSampler

T = TypeVar("T")


def _shared_random_seed() -> int:
    seed = torch.randint(0, 2 ** 31, size=())
    if torch.distributed.is_initialized():  # type: ignore
        if torch.distributed.get_backend() == "nccl":  # type: ignore
            seed = seed.cuda()
        torch.distributed.broadcast(seed, 0)  # type: ignore
    return int(seed)


class DistributedSubsetSampler(Sampler[int], Generic[T]):
    def __init__(
        self,
        dataset: Sequence[T],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        if num_replicas is None:
            num_replicas = torch.distributed.get_world_size()  # type: ignore
        if rank is None:
            rank = torch.distributed.get_rank()  # type: ignore
        if seed is None:
            seed = _shared_random_seed()

        indices = list(range(len(dataset)))
        rng = np.random.RandomState(seed)
        if shuffle:
            rng.shuffle(indices)

        n_total_samples = len(dataset)
        n_sub_samples = (n_total_samples + num_replicas - 1) // num_replicas
        b = n_total_samples * rank // num_replicas
        e = b + n_sub_samples
        self.indices = indices[b:e]

        if shuffle:
            self._sampler: Optional[SubsetRandomSampler] = SubsetRandomSampler(
                self.indices
            )
        else:
            self._sampler = None

    def __iter__(self) -> Iterator[int]:
        if self._sampler is not None:
            return self._sampler.__iter__()  # type: ignore
        else:
            return iter(self.indices)

    def __len__(self) -> int:
        if self._sampler is not None:
            return int(self._sampler.__len__())  # type: ignore
        else:
            return len(self.indices)
