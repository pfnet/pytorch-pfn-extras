# Download MNIST dataset for examples.

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_error_handler.ps1"

New-Item ../data/MNIST/raw -ItemType Directory
Push-Location ../data/MNIST/raw
gsutil -m cp -r "gs://chainer-artifacts-pfn-public-ci/pytorch-pfn-extras/mnist/*" .
Pop-Location
