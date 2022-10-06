import marko
import onnx
import torch
import re

from collections import OrderedDict
from typing import List, Set, Tuple


_scope_re = re.compile("(.+), scope: ([^ ]+)")
_const_vals_re = re.compile(r"value= ([\d\- ]+) \[ \w+Type\{\d+\} \]")
_const_typed_val_re = re.compile(r"value=\[ \w+Type\{(-?[\d\.e-]+)\} \]")
_const_val_re = re.compile(r"value=\{(-?[\d\.e-]+)\}")
_func_re = re.compile(r" = \^(\w+)\(")


class ReconstructError(Exception):
    pass


def _process_line(line: str) -> Tuple[str, str]:
    scope_match = re.match(_scope_re, line)
    scope = ""
    if scope_match is not None:
        scope = scope_match[2].split("/")[-1]
        line = scope_match[1]
    line = line.replace("onnx::Constant", "prim::Constant")
    line = line.replace("onnx::SequenceConstruct", "prim::ListConstruct")
    if "prim::Constant" in line:
        line = re.sub(_const_vals_re, lambda m: f"value=[{m[1].replace('  ', ', ')}]", line)
        line = re.sub(_const_typed_val_re, r"value=\1", line)
        line = re.sub(_const_val_re, r"value=\1", line)

    func_match = re.search(_func_re, line)
    if func_match:
        raise ReconstructError(f"torch.autograd.Function call not supported for: {func_match[1]} in line: {line}")

    return line, scope


def _process_markdown(md: str) -> Tuple[List[str], List[str]]:
    lines: List[str] = []
    scopes: List[str] = []
    target_para: bool = False
    for c in marko.parser.Parser().parse(md).children:  # type: ignore[union-attr]
        if isinstance(c, marko.block.FencedCode) and target_para:
            for text in c.children:
                if not isinstance(text, marko.inline.RawText):
                    continue
                for line in text.children.split("\n"):
                    if len(line) == 0:
                        continue
                    line, scope = _process_line(line)
                    lines.append(line)
                    scopes.append(scope)
            target_para = False
            break
        if not isinstance(c, marko.block.Heading) or c.level != 2:
            continue
        if c.children[0].children == "Original node":
            target_para = True

    return lines, scopes


def reconstruct(model: onnx.ModelProto) -> Tuple[torch._C.Graph, List[Tuple[str, torch.Tensor]]]:
    lines: List[str] = []
    scopes: List[str] = []
    for n in model.graph.node:
        if len(n.doc_string) == 0 and n.op_type != "Constant":
            raise ReconstructError(f"doc_string not found in node: {onnx.helper.printable_node(n)}. Please use strip_doc_string=False option")
        new_lines, new_scopes = _process_markdown(n.doc_string)
        lines.extend(new_lines)
        scopes.extend(new_scopes)
    lines = list(OrderedDict.fromkeys(lines))

    skip_inputs: Set[str] = set([i.name for i in model.graph.initializer])

    inputs: List[str] = ["%" + i.name for i in model.graph.input if i.name not in skip_inputs]
    outputs: List[str] = ["%" + o.name.split(".")[-1] for o in model.graph.output]
    body = "\n    ".join(lines)

    initializer_name_re = re.compile(r"^%([\w.]+) [:=]")
    params: List[Tuple[str, torch.Tensor]] = []
    for i in model.graph.initializer:
        i_name = re.match(initializer_name_re, i.doc_string)
        if i_name:
            inputs.append(f"%{i_name[1]}")
            params.append((i.name, torch.from_numpy(onnx.numpy_helper.to_array(i).copy())))

    src: str = f"""graph({", ".join(inputs)}):
    {body}
    return ({", ".join(outputs)})
"""

    g: torch._C.Graph = torch._C.parse_ir(src)
    torch._C._jit_pass_lint(g)

    return g, params
