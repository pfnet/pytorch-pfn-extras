import os

import pytest
import torch

import pytorch_pfn_extras.onnx as tou
from tests.pytorch_pfn_extras_tests.onnx_tests.test_export_testcase import Net


@pytest.mark.filterwarnings("ignore:Named tensors .* experimental:UserWarning")
def test_onnx_load_model():
    model = Net()
    outdir = "out/load_model_test"
    tou.export_testcase(model, torch.rand(1, 1, 28, 28), outdir,
                        training=True, do_constant_folding=False)
    tou.load_model(os.path.join(outdir, "model.onnx"))


@pytest.mark.filterwarnings("ignore:.*ONNX contains stripped .*:UserWarning")
def test_stripped_onnx_load_model():
    model = Net()
    outdir = "out/stripped_load_model_test"
    tou.export_testcase(model, torch.rand(1, 1, 28, 28), outdir,
                        strip_large_tensor_data=True, training=True,
                        do_constant_folding=False)
    tou.load_model(os.path.join(outdir, "model.onnx"))
