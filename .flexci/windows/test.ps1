Param(
    [String]$test
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_error_handler.ps1"

# Requirements
if ($test == "torch17") {
    RunOrDie python -m pip install torch===1.7.* torchvision===0.8.* -f https://download.pytorch.org/whl/torch_stable.html
} else if ($test == "torch16") {
    RunOrDie python -m pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
} else if ($test == "torch15") {
    RunOrDie python -m pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html
} else {
    throw "Unsupported test variant: $test"
}
RunOrDie python -m pip install pytorch-ignite pytest flake8 matplotlib tensorboard onnx ipython ipywidgets pandas optuna cupy-cuda102

# Install
RunOrDie python -m pip install -e .

# Unit Test
RunOrDie python -m pytest tests

# Examples
RunOrDie python example/mnist.py --batch-size 2048 --test-batch-size 2048 --epochs 1 --save-model
RunOrDie python example/ignite-mnist.py --batch_size 2048 --val_batch_size 2048 --epochs 1
