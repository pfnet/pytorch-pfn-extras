from pytorch_pfn_extras.distributed import create_distributed_subset_indices


def test_not_shuffle() -> None:
    indices0 = create_distributed_subset_indices(
        num_total_samples=10,
        num_replicas=3,
        rank=0,
        shuffle=False,
    )
    assert indices0 == [0, 1, 2, 3]

    indices1 = create_distributed_subset_indices(
        num_total_samples=10,
        num_replicas=3,
        rank=1,
        shuffle=False,
    )
    assert indices1 == [3, 4, 5, 6]

    indices1 = create_distributed_subset_indices(
        num_total_samples=10,
        num_replicas=3,
        rank=2,
        shuffle=False,
    )
    assert indices1 == [6, 7, 8, 9]


def test_shuffle() -> None:
    indices0 = create_distributed_subset_indices(
        num_total_samples=10, num_replicas=3, rank=0, shuffle=True, seed=0
    )
    indices1 = create_distributed_subset_indices(
        num_total_samples=10, num_replicas=3, rank=1, shuffle=True, seed=0
    )
    indices2 = create_distributed_subset_indices(
        num_total_samples=10, num_replicas=3, rank=2, shuffle=True, seed=0
    )
    assert len(indices0 + indices1 + indices2) == 12
