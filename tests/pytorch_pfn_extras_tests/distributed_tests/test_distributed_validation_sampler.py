from unittest import mock

import pytest
import torch.distributed as dist
from pytorch_pfn_extras.distributed import DistributedValidationSampler

_world_size = 4
_dataset_len = 21


@pytest.fixture
def base_dataset():
    return list(range(_dataset_len))


def test_default(base_dataset):
    expected_lengths = [6, 5, 5, 5]
    sample_idxs = []
    with mock.patch.object(
        dist, "get_world_size", return_value=_world_size
    ), mock.patch.object(dist, "is_available", return_value=True):
        for rank in range(_world_size):
            with mock.patch.object(dist, "get_rank", return_value=rank):
                sampler = DistributedValidationSampler(base_dataset)
                assert len(sampler) == expected_lengths[rank]
                sample_idxs += list(sampler)

    # Check no duplication among workers
    assert len(sample_idxs) == len(set(sample_idxs)) == _dataset_len

    # ordered randomly by default
    assert sample_idxs != sorted(sample_idxs)


def test_no_shuffle(base_dataset):
    expected_samples = [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
    ]
    with mock.patch.object(
        dist, "get_world_size", return_value=_world_size
    ), mock.patch.object(dist, "is_available", return_value=True):
        for rank in range(_world_size):
            with mock.patch.object(dist, "get_rank", return_value=rank):
                sampler = DistributedValidationSampler(
                    base_dataset, shuffle=False
                )
                assert list(sampler) == expected_samples[rank]


def test_manual_num_replicas_and_ranks(base_dataset):
    # When manually specifying num_replicas and rank,
    # it doesn't rely on these torch.distributed functions.
    expected_lengths = [6, 5, 5, 5]
    with mock.patch.object(
        dist, "get_world_size", side_effect=AssertionError()
    ), mock.patch.object(
        dist, "is_available", side_effect=AssertionError()
    ), mock.patch.object(
        dist, "get_rank", side_effect=AssertionError()
    ):
        for rank in range(_world_size):
            sampler = DistributedValidationSampler(
                base_dataset, num_replicas=_world_size, rank=rank
            )
            assert len(sampler) == expected_lengths[rank]


def test_seed(base_dataset):
    sampler1 = DistributedValidationSampler(
        base_dataset, num_replicas=_world_size, rank=0, seed=1
    )
    sampler2 = DistributedValidationSampler(
        base_dataset, num_replicas=_world_size, rank=0, seed=2
    )
    assert list(sampler1) != list(sampler2)


def test_no_distributed_available(base_dataset):
    with pytest.raises(RuntimeError):
        DistributedValidationSampler(base_dataset, num_replicas=_world_size)
    with pytest.raises(RuntimeError):
        DistributedValidationSampler(base_dataset, rank=0)


def test_invalid_rank(base_dataset):
    with mock.patch.object(dist, "get_world_size", return_value=_world_size):
        with pytest.raises(ValueError):
            DistributedValidationSampler(base_dataset, rank=-1)
        with pytest.raises(ValueError):
            DistributedValidationSampler(base_dataset, rank=_world_size)
