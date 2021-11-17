from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Set, Union

import torch


@dataclass
class CodeBlock:
    func: Callable[[Dict[str, Any]], Dict[str, Any]]
    optimizer: Optional[torch.optim.Optimizer]
    backprop: bool
    backprop_from: Optional[str]
    backprop_to: Optional[Set[str]]
    state: Dict[str, Any]
    runtime: Any

    def state_dict(self) -> Dict[str, Any]:
        state = {}
        for k, v in self.state.items():
            if hasattr(v, "state_dict"):
                state[k] = v.state_dict()
            else:
                state[k] = v
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for key in state.keys():
            if key not in self.state:
                self.state[key] = state[key]
            elif hasattr(self.state[key], "load_state_dict"):
                self.state[key].load_state_dict(state[key])
            else:
                self.state[key] = state[key]

    def __call__(self, inputs: Any) -> Any:
        return self.runtime.execute(self, inputs)


def update_parameters(
    block: Union[Callable, CodeBlock],
    optimizer: torch.optim.Optimizer,
    backprop_from: Optional[str] = None,
    backprop_to: Optional[Set[str]] = None,
) -> CodeBlock:
    if isinstance(block, CodeBlock):
        func = block.func
        state = block.state
        runtime = block.runtime
        assert not block.backprop
    else:
        func = block
        state = {}
        runtime = block._ppe_runtime
    return CodeBlock(
        func,
        optimizer,
        True,
        backprop_from,
        backprop_to,
        state,
        runtime,
    )


def forward(
    block: Union[Callable, CodeBlock],
) -> CodeBlock:
    if isinstance(block, CodeBlock):
        func = block.func
        state = block.state
        runtime = block.runtime
    else:
        func = block
        state = {}
        runtime = block._ppe_runtime
    return CodeBlock(
        func,
        None,
        False,
        None,
        None,
        state,
        runtime,
    )
