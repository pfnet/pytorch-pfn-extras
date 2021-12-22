import pytest
import torch
import torchvision
from flaky import flaky

from tests.pytorch_pfn_extras_tests.onnx_tests.utils import run_model_test


@flaky
def test_eval_resnet18():
    run_model_test(
        torchvision.models.resnet.resnet18(pretrained=True),
        (torch.rand(1, 3, 224, 224),),
        rtol=1e-03,
        use_gpu=True,
    )


@pytest.mark.xfail
def test_train_resnet18():
    run_model_test(
        torchvision.models.resnet.resnet18(pretrained=True),
        (torch.rand(1, 3, 224, 224),),
        rtol=1e-03,
        use_gpu=True,
        mode="train",
    )


def test_shufflenet():
    run_model_test(
        torchvision.models.shufflenetv2.shufflenet_v2_x1_0(),
        (torch.rand(1, 3, 224, 224),),
        use_gpu=True,
    )
