import marko
import onnx
import torch
import re

from collections import OrderedDict
from typing import List, Tuple


_scope_re = re.compile("(.+), scope: ([^ ]+)")
_const_vals_re = re.compile(r"value= ([\d ]+) \[ \w+Type\{\d+\} \]")
_const_val_re = re.compile(r"value=\{(\d+)\}")
_func_re = re.compile(r" = \^(\w+)\(")


def _process_line(line: str) -> (str, str):
    scope_match = re.match(_scope_re, line)
    scope = ""
    if scope_match is not None:
        scope = scope_match[2].split("/")[-1]
        line = scope_match[1]
    line = line.replace("onnx::Constant", "prim::Constant")
    if "prim::Constant" in line:
        line = re.sub(_const_vals_re, lambda m: f"value=[{m[1].replace('  ', ', ')}]", line)
        line = re.sub(_const_val_re, r"value=\1", line)

    func_match = re.search(_func_re, line)
    if func_match:
        raise RuntimeError(f"Function call not supported for: {func_match[1]} in line: {line}")

    return line, scope


def reconstruct(model: onnx.ModelProto) -> Tuple[torch._C.Graph, List[Tuple[str, torch.Tensor]]]:
    original_lines: List[str] = []
    scopes: List[str] = []
    for n in model.graph.node:
        original_paragraph: bool = False
        for c in marko.parser.Parser().parse(n.doc_string).children:
            if isinstance(c, marko.block.FencedCode) and original_paragraph:
                for lines in c.children:
                    if not isinstance(lines, marko.inline.RawText):
                        continue
                    for line in lines.children.split("\n"):
                        if len(line) == 0:
                            continue
                        line, scope = _process_line(line)
                        original_lines.append(line)
                        scopes.append(scope)
                original_paragraph = False
                break
            if not isinstance(c, marko.block.Heading) or c.level != 2:
                continue
            if c.children[0].children == "Original node":
                original_paragraph = True
    original_lines = list(OrderedDict.fromkeys(original_lines))

    inputs: List[str] = ["%" + i.name for i in model.graph.input]
    outputs: List[str] = ["%" + o.name.split(".")[-1] for o in model.graph.output]
    lines: str = "\n    ".join(original_lines)

    initializer_name_re = re.compile(r"^%(\w+) [:=]")
    params: List[Tuple[str, torch.Tensor]] = []
    for i in model.graph.initializer:
        i_name = re.match(initializer_name_re, i.doc_string)
        if i_name:
            inputs.append(f"%{i_name[1]}")
            params.append((i.name, torch.from_numpy(onnx.numpy_helper.to_array(i).copy())))

    src: str = f"""graph({", ".join(inputs)}):
    {lines}
    return ({", ".join(outputs)})
"""

    g: torch._C.Graph = torch._C.parse_ir(src)
    torch._C._jit_pass_lint(g)

    return g, params
