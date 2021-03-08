#!/bin/bash

set -uex

# Install
pip install .

# Show packages
pip list

# Run unit tests
python -m pytest tests/

# Run examples
.flexci/linux/download_mnist.sh
python example/mnist.py --batch-size 2048 --test-batch-size 2048 --epochs 1 --save-model
python example/ignite-mnist.py --batch_size 2048 --val_batch_size 2048 --epochs 1

# Run flake8
flake8 .
