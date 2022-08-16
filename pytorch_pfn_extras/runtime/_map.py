from typing import Any, Callable, Optional, Sequence, Set

import pytorch_pfn_extras as ppe


def map(
    func: Callable[[Any], Any],
    iterable: Sequence[Any],
    out_keys: Optional[Set[str]] = None,
    device: Any = "cpu",
) -> Sequence[Any]:
    codeblock = ppe.handler.forward(func)
    return codeblock.runtime.map(codeblock, iterable, out_keys, device)  # type: ignore
