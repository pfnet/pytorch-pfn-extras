from typing import TypeVar, Optional, Iterator, Sized

import numpy as np
import torch
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)


class DistributedValidationSampler(torch.utils.data.Sampler):
    """Distributed sampler without duplication

    This sampler splits the input dataset to each worker process in distributed setup
    without allowing repetition.
    It is for evaluation purpose such as :class:`~DistributedEvaluator`.
    This does not guarantee each worker to get the same number of samples,
    so for training do not use this sampler (use PyTorch DistributedSampler instead).
    """

    def __init__(self,
                 dataset: Sized,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0) -> None:
        if num_replicas is None:
            if not dist.is_available():  # type: ignore[no-untyped-call]
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()  # type: ignore[no-untyped-call]
        if rank is None:
            if not dist.is_available():  # type: ignore[no-untyped-call]
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()  # type: ignore[no-untyped-call]
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed

        self.dataset_len = len(dataset)
        self.num_samples = len(np.array_split(range(self.dataset_len), num_replicas)[rank])

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = torch.randperm(self.dataset_len, generator=g).tolist()
        else:
            indices = list(range(self.dataset_len))

        return iter(np.array_split(indices, self.num_replicas)[self.rank])

    def __len__(self) -> int:
        return self.num_samples
