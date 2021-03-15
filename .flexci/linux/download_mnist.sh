#!/bin/bash -uex

# Download MNIST dataset for examples.

set -uex
#curl -LO http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
#curl -LO http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
#curl -LO http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
#curl -LO http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
mkdir -p mnist_raw
gsutil -m cp -r "gs://chainer-artifacts-pfn-public-ci/pytorch-pfn-extras/mnist/*" mnist_raw
