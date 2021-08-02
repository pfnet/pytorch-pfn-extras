# pytorch-pfn-extras

[![PyPI](https://img.shields.io/pypi/v/pytorch-pfn-extras)](https://pypi.python.org/pypi/pytorch-pfn-extras)
[![Docs](https://img.shields.io/readthedocs/pytorch-pfn-extras)](https://pytorch-pfn-extras.readthedocs.io/)
[![License](https://img.shields.io/github/license/pfnet/pytorch-pfn-extras)](https://github.com/pfnet/pytorch-pfn-extras/blob/master/LICENSE)

Supplementary components to accelerate research and development in PyTorch.

## Installation

```sh
pip install pytorch-pfn-extras

# Use `[onnx]` to use onnx submodule like:
#  pip install "pytorch-pfn-extras[onnx]"

### Optinal dependencies
# For PlotReport / VariableStatisticsPlot extensions
pip install matplotlib

# For IgniteExtensionsManager
pip install pytorch-ignite torchvision

# For CuPy interoperability (see: https://docs.cupy.dev/en/stable/install.html)
pip install cupy  # or cupy-cudaXXX
```

## Requirements

* Python 3.6+
* PyTorch 1.7+

Optional dependencies:

* CuPy 8.0+ for PyTorch/CuPy interoperatbility

## Documentation

* [Extensions Manager](docs/source/user_guide/extensions.md)
* [Reporting](docs/source/user_guide/reporting.md)
* [Lazy Modules](docs/source/user_guide/lazy.md)
* [Distributed Snapshot](docs/source/user_guide/snapshot.md)
* [Config System](docs/source/user_guide/config.md)
* [ONNX Utils](docs/source/user_guide/onnx.md)
* [CUDA Utils (CuPy Interoperability)](docs/source/user_guide/cuda.md)

## Examples

* [Custom training loop](example/mnist.py)
* [Ignite integration](example/ignite-mnist.py)

## Contribution Guide

You can contribute to this project by sending a pull request.
After approval, the pull request will be merged by the reviewer.

Before making a contribution, please confirm that:

- Code quality stays consistent across the script, module or package.
- Code is covered by unit tests.
- API is maintainable.

## License

MIT License
