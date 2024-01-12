from typing import List, Optional, TypeVar

import torch.distributed

T = TypeVar("T")


def all_gather_object(obj: T) -> List[Optional[T]]:
    world_size = torch.distributed.get_world_size()  # type: ignore
    object_list: List[Optional[T]] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(object_list=object_list, obj=obj)  # type: ignore
    return object_list
