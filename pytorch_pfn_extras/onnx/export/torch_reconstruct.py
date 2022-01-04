import marko
import onnx
import torch

from typing import Dict, List


def reconstruct(model: onnx.ModelProto) -> torch._C.Graph:
    g = torch._C.Graph()
    values: Dict[str, torch._C.Value] = {}
    inputs: List[torch._C.Value] = []
    for i in model.graph.input:
        inputs.append(g.addInput())
        inputs[-1].setDebugName(i.name)
        values[i.name] = inputs[-1]
    original_lines: List[str] = []
    for n in model.graph.node:
        p = marko.parser.Parser()
        md = p.parse(n.doc_string)
        original_paragraph: bool = False
        for c in md.children:
            if isinstance(c, marko.block.Paragraph) and original_paragraph:
                lines = [line.children for line in c.children if isinstance(line, marko.inline.RawText)]
                print(lines)
                original_lines.extend(lines)
                original_paragraph = False
            if not isinstance(c, marko.block.Heading):
                continue
            if c.level != 2:
                continue
            if c.children[0].children == "Original node":
                original_paragraph = True
    original_lines = list(set(original_lines))
    print(original_lines)

    assert len(list(g.nodes())) == len(model.graph.node)

    return g
