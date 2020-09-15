#!/bin/bash -uex

ln -s /opt/conda/bin/pip /opt/conda/bin/pip3
# torch & torchvision is already installed.
pip3 install pytorch-ignite pytest flake8 matplotlib tensorboard onnx ipython ipywidgets
# TODO(kmaehashi): fix to use stable version after v8 release
pip3 install 'cupy-cuda101>=8.0.0rc1'
pip3 install -e .

# Run unit tests
python3 -m pytest tests/

# Run examples
python3 example/mnist.py --batch-size 2048 --test-batch-size 2048 --epochs 1 --save-model
python3 example/ignite-mnist.py --batch_size 2048 --val_batch_size 2048 --epochs 1

# Run flake8
flake8 .
