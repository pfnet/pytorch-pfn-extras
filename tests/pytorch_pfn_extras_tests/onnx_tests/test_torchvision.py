import pytest
import torch
import torchvision

import pytorch_pfn_extras
from pytorch_pfn_extras_tests.onnx_tests.utils import run_model_test


if pytorch_pfn_extras.requires("1.12.0"):
    resnet18_kwargs = {'weights': None}
else:
    resnet18_kwargs = {'pretrained': True}


def test_eval_resnet18():
    torch.manual_seed(100)
    run_model_test(
        torchvision.models.resnet.resnet18(**resnet18_kwargs),
        (torch.rand(1, 3, 224, 224),),
        rtol=1e-03,
        use_gpu=True,
    )


@pytest.mark.gpu
@pytest.mark.xfail
def test_train_resnet18():
    run_model_test(
        torchvision.models.resnet.resnet18(**resnet18_kwargs),
        (torch.rand(1, 3, 224, 224),),
        rtol=1e-03,
        use_gpu=True,
        mode="train",
    )


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:__floordiv__ is deprecated:UserWarning")
def test_shufflenet():
    run_model_test(
        torchvision.models.shufflenetv2.shufflenet_v2_x1_0(),
        (torch.rand(1, 3, 224, 224),),
        use_gpu=True,
    )
