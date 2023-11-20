import pytest
import torch
import torchvision

import pytorch_pfn_extras
from pytorch_pfn_extras_tests.onnx_tests.utils import run_model_test


if pytorch_pfn_extras.requires("1.12.0"):
    resnet18_kwargs = {'weights': None}
else:
    resnet18_kwargs = {'pretrained': True}


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


@pytest.mark.filterwarnings("ignore:torch.onnx.dynamo_export only implements opset version 18 for now. If you need to use a different opset version, please register them with register_custom_op.:UserWarning")
@pytest.mark.filterwarnings("ignore:Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule! Call GraphModule.add_submodule to add the necessary submodule, GraphModule.add_parameter to add the necessary Parameter, or nn.Module.register_buffer to add the necessary buffer:UserWarning")
@pytest.mark.filterwarnings("ignore:.*does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target:UserWarning")
def test_eval_resnet18_dynamo():
    if not pytorch_pfn_extras.requires("2.1.0"):
        pytest.skip()

    m = torchvision.models.resnet.resnet18(**resnet18_kwargs)
    m.eval()
    run_model_test(
        m,
        (torch.rand(1, 3, 224, 224),),
        rtol=1e-03,
        use_gpu=True,
        use_dynamo=True,
        opset_version=18,
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
