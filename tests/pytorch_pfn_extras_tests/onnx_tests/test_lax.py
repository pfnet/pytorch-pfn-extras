import os
import sys

import onnxruntime as ort
import pytest
import pytorch_pfn_extras
import torch
import torch.nn as nn
import torch.onnx

from pytorch_pfn_extras.onnx import lax
from pytorch_pfn_extras_tests.onnx_tests.test_export_testcase import _helper


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_fori_loop_no_export():
    if not pytorch_pfn_extras.requires("1.8.0"):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    torch.manual_seed(0)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Linear(1, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x * 0.5
            h = lax.fori_loop(1, 4, lambda it, val: it * self.relu(self.linear(val)), x)
            return h + 1

    model = Net()
    x = torch.ones((1, 1))
    y = model(x)
    v = x * 0.5
    for i in range(1, 4):
        v = i * model.relu(model.linear(v))
    y_expected = v + 1
    torch.testing.assert_close(y, y_expected)


@pytest.mark.filterwarnings("ignore:The shape inference of ai.onnx.preview..Gradient type is missing:UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_fori_loop():
    if not pytorch_pfn_extras.requires('1.8.0'):
        pytest.skip('skip for PyTorch 1.7 or earlier')

    if pytorch_pfn_extras.requires('1.10.0') and sys.platform == 'win32':
        pytest.skip('ONNX grad test does not work in windows CI for torch >= 1.10')

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = nn.Linear(1, 1)
            self.linear.weight.data[:] = 2
            self.linear.bias.data[:] = 3

        def forward(self, x):
            x = x * 0.5
            h = lax.fori_loop(1, 4, lambda it, val: it * self.linear(val) + 0.5, x)
            return h + 1

    model = Net()
    x = torch.ones(1, 1)
    output_dir = _helper(
        model,
        x,
        'fori_loop',
        enable_onnx_checker=False,
        use_pfto=False,
        do_constant_folding=False,
    )

    ort_session = ort.InferenceSession(os.path.join(output_dir, "model.onnx"))
    actual = ort_session.run(None, {"input_0": x.cpu().numpy()})
    expected = model(x)
    torch.testing.assert_close(expected, torch.tensor(actual[0]))
