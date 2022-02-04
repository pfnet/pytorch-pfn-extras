import os
import setuptools


here = os.path.abspath(os.path.dirname(__file__))
# Get __version__ variable
exec(open(os.path.join(here, 'pytorch_pfn_extras', '_version.py')).read())


setuptools.setup(
    name='pytorch-pfn-extras',
    version=__version__,        # NOQA
    description='Supplementary components to accelerate research and '
                'development in PyTorch.',
    author='Preferred Networks, Inc.',
    license='MIT License',
    install_requires=['numpy', 'packaging', 'torch', 'typing-extensions>=3.10'],
    extras_require={
        'test': ['pytest', 'onnxruntime', 'torchvision'],
        'onnx': ['onnx'],
    },
    python_requires='>=3.6.0',
    packages=setuptools.find_packages(exclude=['tests', 'tests.*']),
    package_data={'pytorch_pfn_extras': ['py.typed']},
)
