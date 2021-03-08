#!/bin/bash -uex

# Download MNIST dataset for examples.

set -uex

mkdir -p ../data/MNIST/raw
pushd ../data/MNIST/raw
curl -LO http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -LO http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -LO http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -LO http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
