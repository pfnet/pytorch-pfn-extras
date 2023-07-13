from typing import Any, Dict, List, Tuple, Union

import torch


def _compare_states(
    s1: Union[Dict[Any, Any], List[Any], Tuple[Any]],
    s2: Union[Dict[Any, Any], List[Any], Tuple[Any]],
    strict: bool = False,
) -> bool:
    def allclose(a: torch.Tensor, b: torch.Tensor) -> bool:
        if strict:
            return bool((a == b).all())
        else:
            return torch.allclose(a, b)

    if isinstance(s1, dict):
        keys = list(s1.keys())
        assert isinstance(s2, dict)
        if set(keys) != set(s2.keys()):
            return False
    elif isinstance(s1, (list, tuple)):
        keys = list(range(len(s1)))
        if len(s1) != len(s2):
            return False

    all_equal = True
    for k in keys:
        if isinstance(s1[k], dict):
            if not isinstance(s2[k], dict):
                return False
            all_equal = all_equal and _compare_states(s1[k], s2[k])
        elif isinstance(s1[k], (list, tuple)):
            if not isinstance(s2[k], (list, tuple)):
                return False
            all_equal = all_equal and _compare_states(s1[k], s2[k])
        elif isinstance(s1[k], torch.Tensor):
            all_equal = all_equal and allclose(s1[k], s2[k])
        else:
            all_equal = all_equal and s1[k] == s2[k]
        if not all_equal:
            return all_equal
    return all_equal
