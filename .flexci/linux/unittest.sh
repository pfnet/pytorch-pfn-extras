#!/bin/bash

set -uex

# Install
pip install .

# Show packages
pip list

# Run unit tests
pushd tests
python -m pytest --cov-report=html --cov pytorch_pfn_extras .
popd

# Run examples
if [ -d mnist_raw ]; then
    mkdir -p data/MNIST/raw
    mv mnist_raw/* data/MNIST/raw
fi
pushd example
python mnist.py --batch-size 2048 --test-batch-size 2048 --epochs 1 --save-model
python mnist_trainer.py --batch-size 2048 --test-batch-size 2048 --epochs 1
python ignite-mnist.py --batch_size 2048 --val_batch_size 2048 --epochs 1
MASTER_ADDR=127.0.0.1 MASTER_PORT=1236 mpirun -n 2 --allow-run-as-root python mnist_ddp.py --batch-size 2048 --test-batch-size 2048 --epochs 1
popd

# Trainer
pushd example
python mnist_custom_logic.py --batch-size 2048 --test-batch-size 2048 --epochs 1
popd

# Comparer
pushd example
mkdir -p comp_dump_cpu
python mnist_trainer.py --device cpu --epochs 1 --batch-size 1024 --deterministic --compare-dump comp_dump_cpu
CUBLAS_WORKSPACE_CONFIG=:4096:8 python mnist_trainer.py --device cuda --epochs 1 --batch-size 1024 --deterministic --compare-with comp_dump_cpu
popd

# Publish coverage report
mv tests/htmlcov /output/htmlcov
