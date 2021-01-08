$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_error_handler.ps1"

# Requirements
RunOrDie python -m pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RunOrDie python -m pip install pytorch-ignite pytest flake8 matplotlib tensorboard onnx ipython ipywidgets pandas optuna cupy-cuda110

# Install
RunOrDie python -m pip install -e .

# Unit Test
RunOrDie python -m pytest tests

# Examples
RunOrDie python example/mnist.py --batch-size 2048 --test-batch-size 2048 --epochs 1 --save-model
RunOrDie python example/ignite-mnist.py --batch_size 2048 --val_batch_size 2048 --epochs 1
