import torch

from pytorch_pfn_extras.samplers import DistributedSubsetSampler


def test_not_shuffle_sampler() -> None:
    dataset = list(torch.arange(10))
    # rank=0: [0, 1, 2, 3]
    # rank=1: [3, 4, 5, 6]
    # rank=2: [6, 7, 8, 9]
    sampler = DistributedSubsetSampler[torch.Tensor](
        dataset, num_replicas=3, rank=0, shuffle=False
    )
    assert list(iter(sampler)) == [0, 1, 2, 3]
    assert len(sampler) == 4

    sampler = DistributedSubsetSampler[torch.Tensor](
        dataset, num_replicas=3, rank=1, shuffle=False
    )
    assert list(iter(sampler)) == [3, 4, 5, 6]
    assert len(sampler) == 4

    sampler = DistributedSubsetSampler[torch.Tensor](
        dataset, num_replicas=3, rank=2, shuffle=False
    )
    assert list(iter(sampler)) == [6, 7, 8, 9]
    assert len(sampler) == 4


def test_shuffle_sampler() -> None:
    dataset = list(torch.arange(10))
    out = []
    sampler = DistributedSubsetSampler[torch.Tensor](
        dataset, num_replicas=3, rank=0, shuffle=True, seed=0
    )
    out.extend(list(iter(sampler)))
    sampler = DistributedSubsetSampler[torch.Tensor](
        dataset, num_replicas=3, rank=1, shuffle=True, seed=0
    )
    out.extend(list(iter(sampler)))
    sampler = DistributedSubsetSampler[torch.Tensor](
        dataset, num_replicas=3, rank=2, shuffle=True, seed=0
    )
    out.extend(list(iter(sampler)))

    assert len(out) == 12
    assert len(set(out)) == len(dataset)
