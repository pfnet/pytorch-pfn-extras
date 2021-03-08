# Download MNIST dataset for examples.

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\_error_handler.ps1"

New-Item ../data/MNIST/raw -ItemType Directory
Push-Location ../data/MNIST/raw
curl.exe -LO http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl.exe -LO http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl.exe -LO http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl.exe -LO http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Pop-Location
