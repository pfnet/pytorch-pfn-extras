#!/bin/bash

set -uex

# Install
pip install .

# Show packages
pip list

# Run unit tests
python -m pytest --cov-report=html --cov pytorch_pfn_extras tests/

# Run examples
if [ -d mnist_raw ]; then
    mkdir -p ../data/MNIST/raw
    mv mnist_raw/* ../data/MNIST/raw
fi
python example/mnist.py --batch-size 2048 --test-batch-size 2048 --epochs 1 --save-model
python example/ignite-mnist.py --batch_size 2048 --val_batch_size 2048 --epochs 1
MASTER_ADDR=127.0.0.1 MASTER_PORT=1236 mpirun -n 2 --allow-run-as-root python example/mnist_ddp.py --batch-size 2048 --test-batch-size 2048 --epochs 1


# Run pysen
pysen run lint 2> /output/pysen.txt || true

# Run flake8
pysen generate .
flake8 .

mv htmlcov /output/htmlcov
