Param(
    [String]$test
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_error_handler.ps1"

. "$PSScriptRoot\_flexci.ps1"


# Setup environment
if ($test -eq "torch110") {
    # PyTorch 1.10 + Python 3.8
    ActivateCUDA 11.8
    ActivatePython 3.8
    RunOrDie python -m pip install -U pip "setuptools<59.6"
    RunOrDieWithRetry 3 python -m pip install torch==1.10.* torchvision==0.11.* -f https://download.pytorch.org/whl/cu113/torch_stable.html
    # NOTE(linsho):
    # Due to reports of the ONNX build for Windows being broken, we are temporarily limiting the version of ONNX.
    # cf. https://github.com/onnx/onnx/issues/6267
    RunOrDie python -m pip install -r tests/requirements.txt cupy-cuda11x "onnx<=1.16.1"

} elseif ($test -eq "torch111") {
    # PyTorch 1.11 + Python 3.9
    ActivateCUDA 11.3
    ActivatePython 3.9
    RunOrDie python -m pip install -U pip "setuptools<59.6"
    RunOrDieWithRetry 3 python -m pip install torch==1.11.* torchvision==0.12.* -f https://download.pytorch.org/whl/cu113/torch_stable.html
    # NOTE(linsho):
    # Due to reports of the ONNX build for Windows being broken, we are temporarily limiting the version of ONNX.
    # cf. https://github.com/onnx/onnx/issues/6267
    RunOrDie python -m pip install -r tests/requirements.txt cupy-cuda11x "onnx<=1.16.1"

} elseif ($test -eq "torch112") {
    # PyTorch 1.12 + Python 3.10
    ActivateCUDA 11.3
    ActivatePython 3.10
    RunOrDie python -m pip install -U pip "setuptools<59.6"
    RunOrDieWithRetry 3 python -m pip install torch==1.12.* torchvision==0.13.* -f https://download.pytorch.org/whl/cu113/torch_stable.html
    # NOTE(linsho):
    # Due to reports of the ONNX build for Windows being broken, we are temporarily limiting the version of ONNX.
    # cf. https://github.com/onnx/onnx/issues/6267
    RunOrDie python -m pip install -r tests/requirements.txt cupy-cuda11x "onnx<=1.16.1"

} elseif ($test -eq "torch113") {
    # PyTorch 1.13 + Python 3.10
    ActivateCUDA 11.7
    ActivatePython 3.10
    RunOrDie python -m pip install -U pip "setuptools<59.6"
    RunOrDieWithRetry 3 python -m pip install torch==1.13.* torchvision==0.14.* -f https://download.pytorch.org/whl/cu117/torch_stable.html
    # NOTE(linsho):
    # Due to reports of the ONNX build for Windows being broken, we are temporarily limiting the version of ONNX.
    # cf. https://github.com/onnx/onnx/issues/6267
    RunOrDie python -m pip install -r tests/requirements.txt cupy-cuda11x "onnx<=1.16.1"

} elseif ($test -eq "torch200") {
    # PyTorch 2.0 + Python 3.10
    ActivateCUDA 11.7
    ActivatePython 3.10
    RunOrDie python -m pip install -U pip "setuptools<59.6"
    RunOrDieWithRetry 3 python -m pip install torch==2.0.* torchvision==0.15.* -f https://download.pytorch.org/whl/cu117/torch_stable.html
    # NOTE(linsho):
    # Due to reports of the ONNX build for Windows being broken, we are temporarily limiting the version of ONNX.
    # cf. https://github.com/onnx/onnx/issues/6267
    RunOrDie python -m pip install -r tests/requirements.txt cupy-cuda11x "onnx<=1.16.1"

} elseif ($test -eq "torch201") {
    # PyTorch 2.1 + Python 3.10
    ActivateCUDA 11.8
    ActivatePython 3.10
    RunOrDie python -m pip install -U pip "setuptools<59.6"
    RunOrDieWithRetry 3 python -m pip install torch==2.1.* torchvision==0.16.* -f https://download.pytorch.org/whl/cu118/torch_stable.html
    # NOTE(linsho):
    # Due to reports of the ONNX build for Windows being broken, we are temporarily limiting the version of ONNX.
    # cf. https://github.com/onnx/onnx/issues/6267
    RunOrDie python -m pip install -r tests/requirements.txt cupy-cuda11x "onnx<=1.16.1"

} elseif ($test -eq "torch202") {
    # PyTorch 2.2 + Python 3.10
    ActivateCUDA 12.1
    ActivatePython 3.10
    RunOrDie python -m pip install -U pip "setuptools<59.6"
    RunOrDieWithRetry 3 python -m pip install torch==2.2.* torchvision==0.17.* -f https://download.pytorch.org/whl/cu121/torch_stable.html
    # NOTE(linsho):
    # Due to reports of the ONNX build for Windows being broken, we are temporarily limiting the version of ONNX.
    # cf. https://github.com/onnx/onnx/issues/6267
    RunOrDie python -m pip install -r tests/requirements.txt cupy-cuda12x "onnx<=1.16.1"

} else {
    throw "Unsupported test variant: $test"
}

RunOrDie python -V
RunOrDie python -m pip list

# Install
RunOrDie python -m pip install -e .

# Unit Test
$Env:JUPYTER_PLATFORM_DIRS = "1"
RunOrDie python -m pytest -m "not mpi" tests

# Examples
.\.flexci\windows\download_mnist.ps1
RunOrDie python example/mnist.py --batch-size 2048 --test-batch-size 2048 --epochs 1 --save-model
RunOrDie python example/ignite-mnist.py --batch_size 2048 --val_batch_size 2048 --epochs 1
