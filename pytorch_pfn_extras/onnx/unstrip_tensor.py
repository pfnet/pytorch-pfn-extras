import argparse
import json
import math
from pathlib import Path

import onnx
import onnx.helper
import onnx.numpy_helper
import numpy
import pytorch_pfn_extras as ppe

from pytorch_pfn_extras.onnx import strip_large_tensor as strip


def _unstrip_tensor(tensor: onnx.TensorProto) -> None:
    meta_dict = {}
    meta_dict_idx = 0
    for i, external_data in enumerate(tensor.external_data):
        if external_data.key != "location":
            continue
        try:
            external_data_dict = json.loads(external_data.value)
            if external_data_dict.get("type", "") == "stripped":
                meta_dict = external_data_dict
                meta_dict_idx = i
                break
        except ValueError:
            continue
    if not meta_dict:
        return None
    ave = meta_dict.get("average", None)
    var = meta_dict.get("variance", None)
    if ave is None or var is None:
        return None

    if ppe.requires("1.13", "onnx"):
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
    else:
        np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor.data_type]
    dummy_array = numpy.random.normal(ave, math.sqrt(var), tensor.dims).astype(np_dtype)
    dummy_tensor = onnx.numpy_helper.from_array(dummy_array)
    tensor.data_location = onnx.TensorProto.DEFAULT
    tensor.raw_data = dummy_tensor.raw_data
    del tensor.external_data[meta_dict_idx]


def _unstrip_graph(graph: onnx.GraphProto) -> None:
    for init in graph.initializer:
        if not strip._is_stripped_or_set_external(init):
            continue
        _unstrip_tensor(init)

    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                _unstrip_graph(attr.g)


def _unstrip_onnx_from_path(path: Path) -> onnx.GraphProto:
    onnx_graph = onnx.load(str(path), load_external_data=False)
    _unstrip_graph(onnx_graph.graph)
    return onnx_graph


def _unstrip_tensor_from_path(path: Path) -> onnx.TensorProto:
    onnx_tensor = onnx.TensorProto()
    with open(path, "rb") as fp:
        onnx_tensor.ParseFromString(fp.read())
    _unstrip_tensor(onnx_tensor)
    return onnx_tensor


def unstrip(path: str, out_path: str = "") -> None:
    """Unstrip ONNX models and test data(.pb).

    Add tensor(raw data) to the target ONNXs (and test data).
    Values are random following mean and variance written in meta information.

    Args:
        path (str): The target directory path, ONNX file, or Tensor (Protobuf)
            file path.
        out_path (str): Output path to be written.
    """
    target_path = Path(path)
    if not target_path.exists():
        print(f"Error: the target path is not found, {path}")
        return None
    written_path = Path(out_path if out_path else path)

    if target_path.is_dir():
        onnx_model_paths = target_path.glob("*.onnx")
        for onnx_path in onnx_model_paths:
            written_onnx_path = written_path / onnx_path.relative_to(target_path)
            written_onnx_path.parent.mkdir(exist_ok=True)
            _write_onnx(_unstrip_onnx_from_path(onnx_path), written_onnx_path)
        data_pathas = target_path.glob("test_data_set_*/*.pb")
        for data_path in data_pathas:
            written_tensor_path = written_path / data_path.relative_to(target_path)
            written_tensor_path.parent.mkdir(exist_ok=True)
            _write_tensor(_unstrip_tensor_from_path(data_path), written_tensor_path)
        meta = target_path / "meta.json"
        if meta.exists():
            _rewrite_meta(meta, written_path)
        return None

    ext = target_path.suffix
    if ext == ".onnx":
        _write_onnx(_unstrip_onnx_from_path(target_path), written_path)
    elif ext == ".pb":
        _write_tensor(_unstrip_tensor_from_path(target_path), written_path)
    else:
        print(f"Error: the target file is not supported, {path}")


def _write_onnx(onnx_graph: onnx.GraphProto, out: Path) -> None:
    with open(out, "wb") as fp:
        fp.write(onnx_graph.SerializeToString())


def _write_tensor(onnx_tensor: onnx.TensorProto, out: Path) -> None:
    with open(out, "wb") as fp:
        fp.write(onnx_tensor.SerializeToString())


def _rewrite_meta(meta: Path, out: Path) -> None:
    with open(meta) as fp:
        meta_dict = json.load(fp)
    meta_dict["strip_large_tensor_data"] = False
    with open(out / "meta.json", "w") as fp:
        json.dump(meta_dict, fp, indent=2)


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--out_path", default="")
    return parser.parse_args()


def _main() -> None:
    args = _get_args()
    unstrip(args.path, args.out_path)
    return None


if __name__ == "__main__":
    _main()
