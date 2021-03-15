#!/bin/bash -uex

# Download MNIST dataset for examples.

set -uex

mkdir -p ../data/MNIST/raw
pushd ../data/MNIST/raw
gsutil -m cp -r "gs://chainer-artifacts-pfn-public-ci/pytorch-pfn-extras/mnist/*" .
