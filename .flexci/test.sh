#!/bin/bash -uex

perl -pi.bak -e 's|http://archive\.ubuntu\.com/ubuntu/|mirror://mirrors.ubuntu.com/mirrors.txt|g' /etc/apt/sources.list
apt update
apt -y install python3 python3-pip

pip3 install torch torchvision pytorch-ignite pytest flake8 matplotlib tensorboard onnx
pip3 install -e .

# Run unit tests
python3 -m pytest tests/

# Run examples
python3 example/mnist.py --batch-size 2048 --test-batch-size 2048 --epochs 1 --save-model
python3 example/ignite-mnist.py --batch_size 2048 --val_batch_size 2048 --epochs 1

# Run flake8
flake8 .
