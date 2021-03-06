Param(
    [String]$test
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_error_handler.ps1"

. "$PSScriptRoot\_flexci.ps1"

# Setup environment
if ($test -eq "torch15") {
    # PyTorch 1.5 + Python 3.6
    ActivatePython 3.6
    RunOrDie python -m pip install torch==1.5.* torchvision==0.6.* -f https://download.pytorch.org/whl/torch_stable.html

} elseif ($test -eq "torch16") {
    # PyTorch 1.6 + Python 3.7
    ActivatePython 3.6
    RunOrDie python -m pip install torch==1.6.* torchvision==0.7.* -f https://download.pytorch.org/whl/torch_stable.html

} elseif ($test -eq "torch17") {
    # PyTorch 1.7 + Python 3.8
    ActivatePython 3.8
    RunOrDie python -m pip install torch==1.7.* torchvision==0.8.* -f https://download.pytorch.org/whl/torch_stable.html

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
RunOrDie python example/mnist.py --batch-size 2048 --test-batch-size 2048 --epochs 1 --save-model
RunOrDie python example/ignite-mnist.py --batch_size 2048 --val_batch_size 2048 --epochs 1
