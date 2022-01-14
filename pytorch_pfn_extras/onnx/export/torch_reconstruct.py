import marko
import onnx
import torch
import re

from collections import OrderedDict
from typing import List


def reconstruct(model: onnx.ModelProto) -> torch._C.Graph:
    original_lines: List[str] = []
    scopes: List[str] = []
    scope_re = re.compile("(.+), scope: (.+)")
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
                        scope_match = re.match(scope_re, line)
                        if scope_match is not None:
                            scopes.append(scope_match[2])
                            line = scope_match[1]
                        else:
                            scopes.append("")
                        line = line.replace("onnx::Constant", "prim::Constant")
                        original_lines.append(line)
                original_paragraph = False
                break
            if not isinstance(c, marko.block.Heading) or c.level != 2:
                continue
            if c.children[0].children == "Original node":
                original_paragraph = True
    original_lines = list(OrderedDict.fromkeys(original_lines))

    inputs: List[str] = [f"%{i.name}" for i in model.graph.input]
    outputs: List[str] = [f"%{o.name}" for o in model.graph.output]
    lines: str = "\n    ".join(original_lines)

    src: str = f"""graph({", ".join(inputs)}):
    {lines}
    return ({", ".join(outputs)})
"""

    g: torch._C.Graph = torch._C.parse_ir(src)
    torch._C._jit_pass_lint(g)

    return g
