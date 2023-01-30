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
* PyTorch 1.9+

Optional dependencies:

* CuPy 8.0+ for PyTorch/CuPy interoperatbility

## Documentation

Refer to [Read The Docs](https://pytorch-pfn-extras.readthedocs.io/) for the complete documentation.

Below are some quick-links to the most important features of the library.

* [Extensions Manager](https://pytorch-pfn-extras.readthedocs.io/en/latest/user_guide/extensions.html)
* [Reporting](https://pytorch-pfn-extras.readthedocs.io/en/latest/user_guide/reporting.html)
* [Lazy Modules](https://pytorch-pfn-extras.readthedocs.io/en/latest/user_guide/lazy.html)
* [Distributed Snapshot](https://pytorch-pfn-extras.readthedocs.io/en/latest/user_guide/snapshot.html)
* [Config System](https://pytorch-pfn-extras.readthedocs.io/en/latest/user_guide/config.html)
* [ONNX Utils](https://pytorch-pfn-extras.readthedocs.io/en/latest/user_guide/onnx.html)
* [CUDA Utils (CuPy Interoperability)](https://pytorch-pfn-extras.readthedocs.io/en/latest/user_guide/cuda.html)

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
