#!/bin/bash

set -uex

# Install
pip install .

# Show packages
pip list

# Run unit tests
python -m pytest tests/

# Run examples
if [ -d mnist_raw ]; then
    mkdir -p ../data/MNIST/raw
    mv mnist_raw/* ../data/MNIST/raw
fi
python example/mnist.py --batch-size 2048 --test-batch-size 2048 --epochs 1 --save-model
python example/ignite-mnist.py --batch_size 2048 --val_batch_size 2048 --epochs 1

# Run pysen
pysen run lint || true

# Run flake8
flake8 .
