Param(
    [String]$test
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_error_handler.ps1"

. "$PSScriptRoot\_flexci.ps1"


# Setup environment
if ($test -eq "torch18") {
    # PyTorch 1.8 + Python 3.7
    ActivateCUDA 11.1
    ActivatePython 3.7
    RunOrDie python -m pip install -U pip "setuptools<59.6"
    RunOrDieWithRetry 3 python -m pip install torch==1.8.* torchvision==0.9.* -f https://download.pytorch.org/whl/cu111/torch_stable.html

} elseif ($test -eq "torch19") {
    # PyTorch 1.9 + Python 3.8
    ActivateCUDA 11.1
    ActivatePython 3.8
    RunOrDie python -m pip install -U pip "setuptools<59.6"
    RunOrDieWithRetry 3 python -m pip install torch==1.9.* torchvision==0.10.* -f https://download.pytorch.org/whl/cu111/torch_stable.html

} elseif ($test -eq "torch110") {
    # PyTorch 1.10 + Python 3.9
    ActivateCUDA 11.3
    ActivatePython 3.9
    RunOrDie python -m pip install -U pip setuptools
    RunOrDieWithRetry 3 python -m pip install torch==1.10.* torchvision==0.11.* -f https://download.pytorch.org/whl/cu113/torch_stable.html

} else {
    throw "Unsupported test variant: $test"
}
RunOrDie python -V

# Install common requirements
RunOrDie python -m pip install pytorch-ignite pytest flake8 matplotlib tensorboard onnx ipython ipywidgets pandas optuna cupy-cuda102
RunOrDie python -m pip list

# Install
RunOrDie python -m pip install -e .

# Unit Test
RunOrDie python -m pytest tests

# Examples
.\.flexci\windows\download_mnist.ps1
RunOrDie python example/mnist.py --batch-size 2048 --test-batch-size 2048 --epochs 1 --save-model
RunOrDie python example/ignite-mnist.py --batch_size 2048 --val_batch_size 2048 --epochs 1
