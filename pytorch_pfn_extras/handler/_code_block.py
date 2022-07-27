from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

import torch


@dataclass
class CodeBlock:
    """Class that is used to specify and apply actions to a callable.

    CodeBlocks are used in Logic classes to write device agnostic codes, as
    the device runtime is in charge of doing the execution of the module with
    the actions requested from the codeblock

    Args:
       func: The function to be operated according to the specified options.
       optimizer: The Optimizer that will be used for parameter update.
       backprop: Flag to specify if gradients are to be calculated.
       backprop_from: Select a single output from the block execution to perform
           the gradient calculation.
       backprop_to: Name of the values where backpropagation will be stopped.
       state: Data that can be used during the CodeBlock execution.
    """
    func: Callable
    optimizers: List[torch.optim.Optimizer]
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
    block: Callable,
    optimizers: List[torch.optim.Optimizer],
    backprop_from: Optional[str] = None,
    backprop_to: Optional[Set[str]] = None,
) -> CodeBlock:
    """
    Returns a ``CodeBlock`` that performs the forward, backward passes and
    applies the optimizer step for the given ``torch.nn.Module`` or another
    ``CodeBlock``.

    Args:
       block: ``torch.nn.Module`` or ``CodeBlock`` to update the parameters.
       optimizers: The list of Optimizer that will be used for parameter update.
       backprop_from: Select a single output from the block execution to perform
           the gradient calculation.
       backprop_to: Name of the values where backpropagation will be stopped.

    Returns: A ``CodeBlock`` object.
    """
    if isinstance(block, CodeBlock):
        assert not block.backprop
    codeblock = forward(block)
    return CodeBlock(
        func=codeblock.func,
        optimizers=optimizers,
        backprop=True,
        backprop_from=backprop_from,
        backprop_to=backprop_to,
        state=codeblock.state,
        runtime=codeblock.runtime,
    )


def forward(block: Callable) -> CodeBlock:
    """
    Returns a ``CodeBlock`` that performs the forward pass for the given
    ``torch.nn.Module`` or another ``CodeBlock``.

    Args:
       block: ``torch.nn.Module`` or ``CodeBlock`` to update the parameters.

    Returns: A ``CodeBlock`` object.
    """
    if isinstance(block, CodeBlock):
        func = block.func
        state = block.state
        runtime = block.runtime
    else:
        module = None
        if isinstance(block, torch.nn.Module):
            module = block
        else:
            module = getattr(block, '__self__', None)
            assert module is not None
        func = block
        state = {}
        runtime = getattr(module, '_ppe_runtime', None)
        assert runtime is not None

    return CodeBlock(
        func=func,
        optimizers=[],
        backprop=False,
        backprop_from=None,
        backprop_to=None,
        state=state,
        runtime=runtime,
    )
