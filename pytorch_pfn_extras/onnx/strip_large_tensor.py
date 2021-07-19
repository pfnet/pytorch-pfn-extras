import argparse
import json
import operator
from functools import reduce

import numpy

import onnx.numpy_helper
import onnx.external_data_helper

LARGE_TENSOR_DATA_THRESHOLD = 100


def is_large_tensor(tensor, threshold):
    size = reduce(operator.mul, tensor.dims, 1)
    return size > threshold


def _is_stripped(tensor):
    for external_data in tensor.external_data:
        if external_data.key != 'location':
            continue
        try:
            external_value_dict = json.loads(external_data.value)
            return external_value_dict.get('type', '') == 'stripped'
        except ValueError:
            # Invalid JSON, indicating `external_data.value` contains
            # a file path
            continue
    return False


def _strip_raw_data(tensor):
    arr = onnx.numpy_helper.to_array(tensor)
    meta_dict = {}
    meta_dict['type'] = "stripped"
    meta_dict['average'] = float(numpy.average(arr))
    meta_dict['variance'] = float(numpy.var(arr))
    onnx.external_data_helper.set_external_data(tensor,
                                                location=json.dumps(meta_dict),
                                                length=len(tensor.raw_data))
    tensor.data_location = onnx.TensorProto.EXTERNAL
    tensor.ClearField('raw_data')
    tensor.ClearField('float_data')
    return tensor


def _strip_large_initializer_raw_data(onnx_model, large_tensor_threshold):
    for init in onnx_model.graph.initializer:
        if _is_stripped(init):
            continue
        if is_large_tensor(init, large_tensor_threshold):
            _strip_raw_data(init)


def _strip_large_tensor_tool_impl(onnx_path, out_onnx_path,
                                  large_tensor_threshold):
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    _strip_large_initializer_raw_data(onnx_model, large_tensor_threshold)
    with open(out_onnx_path, 'wb') as fp:
        fp.write(onnx_model.SerializeToString())


def _strip_large_tensor_tool():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_path', type=str)
    parser.add_argument('--out_onnx_path', type=str, default=None)
    parser.add_argument('--large_tensor_threshold',
                        type=int,
                        default=LARGE_TENSOR_DATA_THRESHOLD)
    args = parser.parse_args()
    out_onnx_path = args.out_onnx_path
    if out_onnx_path is None:
        out_onnx_path = args.onnx_path

    _strip_large_tensor_tool_impl(args.onnx_path, out_onnx_path,
                                  args.large_tensor_threshold)


if __name__ == '__main__':
    _strip_large_tensor_tool()
