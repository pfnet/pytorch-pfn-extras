import onnx
from typing import Any, Generator, List, NamedTuple, Tuple
import torch
import threading
from contextlib import contextmanager

_outputs = threading.local()


class _Output(NamedTuple):
    name: str
    value: torch.Tensor


class _Outputs:
    _values: List[_Output]

    def __init__(self) -> None:
        self._values = []

    def clear(self) -> None:
        self._values.clear()

    @property
    def values(self) -> List[_Output]:
        return self._values

    def add(self, name: str, value: torch.Tensor) -> None:
        self._values.append(_Output(name, value))

    def add_outputs_to_model(self, onnx_graph: onnx.ModelProto) -> None:
        if len(self.values) == 0:
            return

        old_name_to_new_name = {}
        n_output = len(onnx_graph.graph.output)
        assert n_output >= len(self.values)

        # Rename last len(self.values) outputs
        for i, additional_output in enumerate(
                onnx_graph.graph.output[-len(self.values):]):
            name = self.values[i].name
            orig_name = additional_output.name
            old_name_to_new_name[orig_name] = name
            additional_output.name = name

        # Rename names in graph
        for node in onnx_graph.graph.node:
            for i, v in enumerate(node.input):
                if v in old_name_to_new_name:
                    node.input[i] = old_name_to_new_name[v]
            for i, v in enumerate(node.output):
                if v in old_name_to_new_name:
                    node.output[i] = old_name_to_new_name[v]

        for v in onnx_graph.graph.input:
            if v.name in old_name_to_new_name:
                v.name = old_name_to_new_name[v.name]


class _ModuleWithAdditionalOutputs(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, outputs: _Outputs) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.module = module
        self.outputs = outputs

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.outputs.clear()
        out = self.module(*args, **kwargs)
        if len(self.outputs.values) == 0:
            return out
        if isinstance(out, torch.Tensor):
            out = [out]
        elif not isinstance(out, list):
            out = list(out)
        out.extend([value for _, value in self.outputs.values])
        return out

    def state_dict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self.module.load_state_dict(*args, **kwargs)


@contextmanager
def trace(
        module: torch.nn.Module
) -> Generator[Tuple[torch.nn.Module, _Outputs], None, None]:
    _outputs.outputs = _Outputs()
    module = _ModuleWithAdditionalOutputs(module, _outputs.outputs)
    try:
        yield module, _outputs.outputs
    finally:
        # onnx_graph = _outputs.add_outputs_to_model(onnx_graph)
        _outputs.outputs = None


def as_output(name: str, value: torch.Tensor) -> torch.Tensor:
    if hasattr(_outputs, "outputs") and _outputs.outputs is not None:
        _outputs.outputs.add(name, value)
    return value
