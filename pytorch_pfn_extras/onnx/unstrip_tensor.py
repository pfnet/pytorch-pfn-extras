import argparse
import json
import math

import onnx
import onnx.numpy_helper
import numpy

from pytorch_pfn_extras.onnx import strip_large_tensor as strip


def _unstrip_tensor(tensor: onnx.TensorProto) -> None:
    meta_dict = {}
    for external_data in tensor.external_data:
        if external_data.key != 'location':
            continue
        try:
            external_data_dict = json.loads(external_data.value)
            if external_data_dict.get('type', '') == 'stripped':
                meta_dict = external_data_dict
                break
        except ValueError:
            continue
    if not meta_dict:
        return None
    ave = meta_dict.get('average', None)
    var = meta_dict.get('variance', None)
    if ave is None or var is None:
        return None

    np_dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor.data_type]
    dummy_arrary = numpy.random.normal(ave, math.sqrt(var), tensor.dims).astype(
        np_dtype)
    dummy_tensor = onnx.numpy_helper.from_array(dummy_arrary)
    tensor.data_location = onnx.TensorProto.DEFAULT
    tensor.raw_data = dummy_tensor.raw_data

    meta_dict['type'] = "unstripped"
    onnx.external_data_helper.set_external_data(
        tensor, location=json.dumps(meta_dict), length=dummy_arrary.nbytes)

    return None


def _unstrip_graph(graph: onnx.GraphProto) -> None:
    for init in graph.initializer:
        if not strip._is_stripped_or_set_external(init):
            continue
        _unstrip_tensor(init)

    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField('g'):
                _unstrip_graph(attr.g)


def unstrip(onnx_path: str, out_onnx_path: str = "") -> None:
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    _unstrip_graph(onnx_model.graph)

    out_path = out_onnx_path if out_onnx_path else onnx_path
    with open(out_path, 'wb') as fp:
        fp.write(onnx_model.SerializeToString())


def _get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_path')
    parser.add_argument('--out_onnx_path', default="")
    return parser.parse_args()


def _main() -> None:
    args = _get_args()
    unstrip(args.onnx_path, args.out_onnx_path)
    return None


if __name__ == '__main__':
    _main()
