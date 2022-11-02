Param(
    [String]$test
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_error_handler.ps1"

. "$PSScriptRoot\_flexci.ps1"


# Setup environment
if ($test -eq "torch19") {
    # PyTorch 1.9 + Python 3.8
    ActivateCUDA 11.1
    ActivatePython 3.8
    RunOrDie python -m pip install -U pip "setuptools<59.6"
    RunOrDieWithRetry 3 python -m pip install torch==1.9.* torchvision==0.10.* -f https://download.pytorch.org/whl/cu111/torch_stable.html
    RunOrDie python -m pip install -U pip "pytorch-ignite==0.4.9"

} elseif ($test -eq "torch110") {
    # PyTorch 1.10 + Python 3.9
    ActivateCUDA 11.3
    ActivatePython 3.9
    RunOrDie python -m pip install -U pip "setuptools<59.6"
    RunOrDieWithRetry 3 python -m pip install torch==1.10.* torchvision==0.11.* -f https://download.pytorch.org/whl/cu113/torch_stable.html

} elseif ($test -eq "torch111") {
    # PyTorch 1.11 + Python 3.9
    ActivateCUDA 11.3
    ActivatePython 3.9
    RunOrDie python -m pip install -U pip "setuptools<59.6"
    RunOrDieWithRetry 3 python -m pip install torch==1.11.* torchvision==0.12.* -f https://download.pytorch.org/whl/cu113/torch_stable.html

} elseif ($test -eq "torch112") {
    # PyTorch 1.12 + Python 3.10
    ActivateCUDA 11.3
    ActivatePython 3.10
    RunOrDie python -m pip install -U pip "setuptools<59.6"
    RunOrDieWithRetry 3 python -m pip install torch==1.12.* torchvision==0.13.* -f https://download.pytorch.org/whl/cu113/torch_stable.html

} elseif ($test -eq "torch113") {
    # PyTorch 1.13 + Python 3.10
    ActivateCUDA 11.7
    ActivatePython 3.10
    RunOrDie python -m pip install -U pip "setuptools<59.6"
    RunOrDieWithRetry 3 python -m pip install torch==1.13.* torchvision==0.14.* -f https://download.pytorch.org/whl/cu117/torch_stable.html

} else {
    throw "Unsupported test variant: $test"
}
RunOrDie python -V

# Install common requirements
RunOrDie python -m pip install pytorch-ignite pytest flake8 matplotlib tensorboard onnx ipython ipywidgets pandas optuna cupy-cuda102 onnxruntime slack_sdk
RunOrDie python -m pip list

# Install
RunOrDie python -m pip install -e .

# Unit Test
RunOrDie python -m pytest tests

# Examples
.\.flexci\windows\download_mnist.ps1
RunOrDie python example/mnist.py --batch-size 2048 --test-batch-size 2048 --epochs 1 --save-model
RunOrDie python example/ignite-mnist.py --batch_size 2048 --val_batch_size 2048 --epochs 1
