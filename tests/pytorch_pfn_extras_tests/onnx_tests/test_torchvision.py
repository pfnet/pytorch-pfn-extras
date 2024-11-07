import pytest
import torch
import torchvision

import pytorch_pfn_extras
from pytorch_pfn_extras_tests.onnx_tests.utils import run_model_test


resnet18_kwargs = {'weights': None}

@pytest.mark.filterwarnings("ignore:Converting a tensor to a Python boolean might cause the trace to be incorrect:torch.jit.TracerWarning")
def test_eval_resnet18():
    old_allow_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = False
        run_model_test(
            torchvision.models.resnet.resnet18(**resnet18_kwargs),
            (torch.rand(1, 3, 224, 224),),
            rtol=1e-03,
            use_gpu=True,
        )
    finally:
        torch.backends.cudnn.allow_tf32 = old_allow_tf32


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
