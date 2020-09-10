# pytorch_pfn_extras.onnx

Extensions to `torch.onnx.export`.

## Installation

```bash
pip3 install "pytorch-pfn-extras[onnx]"
```

Or

1. Install pytorch-pfn-extras normally
2. Install onnx with `pip install onnx==1.7.0`

## API

### `pytorch_pfn_extras.onnx.export_testcase`

Instead of specifying file name in `torch.onnx.export`, `pytorch_pfn_extra.onnx.export_testcase` specifies directory to output ONNX model and test case in/out.

```python
import torch
import torch.nn as nn
model = nn.Sequential(nn.Linear(5, 10, bias=False))
x = torch.zeros((2, 5))

import pytorch_pfn_extras.onnx as tou
tou.export_testcase(model, x, '/path/to/output')
```

Directory structure with following will be generated to `/path/to/output`:

```bash
$ tree /path/to/output
/path/to/output
├── meta.json
├── model.onnx
└── test_data_set_0
    ├── input_0.pb
    └── output_0.pb
```

* This directory structure format is inspired by ONNX official test data set: (Example: [node](https://github.com/onnx/onnx/tree/master/onnx/backend/test/data/node)). PyTorch's ONNX tests use this format too. (Reference: [export_onnx_tests_generator.py](https://github.com/pytorch/pytorch/blob/master/test/onnx/export_onnx_tests_generator.py))
  * There are scripts in [chainer-compiler/utils](https://github.com/pfnet-research/chainer-compiler/tree/master/utils) to run inference in major runtime with the directory structure. For example to inference with ONNXRuntime, run `$ python run_onnx_onnxruntime.py /path/to/output` to use `input_N.pb` as input and compare numerically with its output `output_N.pb`(N is the index of test case).
* By default `meta.json` is generated too to track git infos, date times, etc. Add `metadata=False` argument to suppress this.

#### `out_grad` option

If `out_grad=True` is specified gradient will be dumped too, which is useful for debugging backward. `gradient_N.pb` and `gradient_input_N.pb` would be dumped to test case directory with in/out data.
`gradient_input_N.pb` is the initial value of backward, and it's default value is ones tensor with same shape of output. Use `out_grad` to specify custom initial value (`torch.Tensor` type) for it.

```python
model = nn.Sequential(nn.Linear(5, 10, bias=False))
x = torch.zeros((2, 5))

import pytorch_pfn_extras.onnx as tou
tou.export_testcase(model, x, '/path/to/output', out_grad=True)
```

```bash
$ tree /path/to/output
/path/to/output
├── meta.json
├── model.onnx
└── test_data_set_0
    ├── gradient_0.pb
    ├── gradient_input_0.pb
    ├── input_0.pb
    └── output_0.pb
```

#### `model_overwrite` option

Use `model_overwrite` option to create multiple data set like following:

```python
import pytorch_pfn_extras.onnx as tou
tou.export_testcase(model, x1, '/path/to/output')
tou.export_testcase(model, x2, '/path/to/output', model_overwrite=False)
```

Following is the generated test cases of the above.
`test_data_set_0` is the input`x1` and is its output, `test_data_set_1` is the input `x2` and its output.

```bash
$ tree /path/to/output
├── meta.json
├── model.onnx
├── test_data_set_0
│   ├── input_0.pb
│   └── output_0.pb
└── test_data_set_1
    ├── input_0.pb
    └── output_0.pb
```

#### `strip_large_tensor_data` option

This option strips large tensor in dumped files which is useful to reduce file size in usage such as benchmarking. Not only `model.onnx`, in/out, gradient data would be affected too.
`large_tensor_threshold` could be used to specify threshold of large tensor size.

```python
import torchvision
model = torchvision.models.resnet50(pretrained=True)
x = torch.zeros((1, 3, 224, 224))

import pytorch_pfn_extras.onnx as tou
tou.export_testcase(model, x, '/path/to/output')
tou.export_testcase(model, x, '/path/to/output2', strip_large_tensor_data=True)
```

```bash
$ ls -lh /path/to/output/model.onnx
-rwxrwxrwx 1 user user 98M Jun 24 23:34 /path/to/output/model.onnx
$ ls -lh /path/to/output2/model.onnx
-rwxrwxrwx 1 user user 64K Jun 24 23:34 /path/to/output2/model.onnx
```

This feature could be called from CLI:

```bash
$ python -m pytorch_pfn_extras.onnx.strip_large_tensor resnet50.onnx --out_onnx_path resnet50_slim.onnx
$ ls -lh
-rwxrwxrwx 1 user user 98M Jun 30 09:13 resnet50.onnx
-rwxrwxrwx 1 user user 64K Jun 30 09:16 resnet50_slim.onnx
```

See `$ python -m pytorch_pfn_extras.onnx.strip_large_tensor -h` for help

### `pytorch_pfn_extras.onnx.export`

Function with same interface like `torch.onnx.export`. Unlike `torch.onnx.export`, you can use annotation feature (described below), `strip_large_tensor_data` options, or other `torch.onnx` extensions.

* `strip_large_tensor_data`: Same as `export_testcase`. Useful reducing file sizes.
* `return_output`: Returns output value of model execution. Note: Most output type would be `torch.Tensor`(not `onnx.TensorProto`)

```python
model = nn.Sequential(nn.Linear(5, 10, bias=False))
x = torch.zeros((2, 5))

import io, onnx
bytesio = io.BytesIO()
pytorch_pfn_extras.onnx.export(model, x, bytesio)
onnx_proto = onnx.load(io.BytesIO(bytesio.getvalue()))
```

### `annotate`

Feature to add custom ONNX attribute to specified `nn.Module`.

Notes:
* Annotated ONNX would be invalid ONNX format that doesn't pass check of onnx.checker.check_model.
* Only valid with `pytorch_pfn_extras.onnx.export_testcase` or `pytorch_pfn_extras.onnx.export` export.
* **Only** the first ONNX node of modules like `nn.Linear`, `nn.GroupNorm`, etc. with multiple ONNX node would be annotated
  * For example `nn.Linear` with bias is split to `MatMul` -> `Add` graph. Only `MatMul` would be annotated. This is same in `apply_annotation` (described later) too.
* Use `apply_annotation` instead when the annotation target isn't `nn.Module`.

```python
import pytorch_pfn_extras.onnx as tou

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Conv2d(6, 9, 3)
        self.conv2 = nn.Conv2d(9, 12, 3)
        self.linear = nn.Linear(28, 20)
        self.linear2 = nn.Linear(20, 15)

    def forward(self, x):
        h = self.conv(x)
        with tou.annotate(key='value'):
            h = self.conv2(h)
            h = self.linear(h)
        h = self.linear2(h)
        return h

model = Net()
x = torch.randn((1, 6, 32, 32))
tou.export_testcase(model, x, '/path/to/output')
onnx_proto = onnx.load(os.path.join('/path/to/output, 'model.onnx'))
print(onnx.helper.printable_graph(onnx_proto.graph))
```

```
graph torch-jit-export (
  %input.1[FLOAT, 1x6x32x32]
) initializers (
  %17[FLOAT, 28x20]
  %18[FLOAT, 20x15]
  %conv.bias[FLOAT, 9]
  %conv.weight[FLOAT, 9x6x3x3]
  %conv2.bias[FLOAT, 12]
  %conv2.weight[FLOAT, 12x9x3x3]
  %linear.bias[FLOAT, 20]
  %linear2.bias[FLOAT, 15]
) {
  %9 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]](%input.1, %conv.weight, %conv.bias)
  %10 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], key = 'value', pads = [0, 0, 0, 0], strides = [1, 1]](%9, %conv2.weight, %conv2.bias)
  %12 = MatMul[key = 'value'](%10, %17)
  %13 = Add(%12, %linear.bias)
  %15 = MatMul(%13, %18)
  %16 = Add(%15, %linear2.bias)
  return %16
}
```

In above example `%10 = Conv` and `%12 = MatMul` has `key='value'` attribute annotated.

### `apply_annotation`

This annotates function call instead of annotating it with `with`.

The `annotate` target is `nn.Module`, so `torch.nn.functional` couldn't be annotated

```python
import torch.nn.functional as F
import pytorch_pfn_extras.onnx as tou

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Conv2d(6, 9, 3)
        self.conv2 = nn.Conv2d(9, 12, 3)
        self.linear = nn.Linear(28, 20)
        self.linear2 = nn.Linear(20, 15)

    def forward(self, x):
        h = self.conv(x)
        with tou.annotate(key='value'):
            h = self.conv2(h)
            h = F.relu(h)
            h = self.linear(h)
        h = self.linear2(h)
        return h

model = Net()
x = torch.randn((1, 6, 32, 32))
tou.export_testcase(model, x, '/path/to/output')
onnx_proto = onnx.load(os.path.join('/path/to/output', 'model.onnx'))
print(onnx.helper.printable_graph(onnx_proto.graph))
```

```
graph torch-jit-export (
  %input.1[FLOAT, 1x6x32x32]
) initializers (
  %18[FLOAT, 28x20]
  %19[FLOAT, 20x15]
  %conv.bias[FLOAT, 9]
  %conv.weight[FLOAT, 9x6x3x3]
  %conv2.bias[FLOAT, 12]
  %conv2.weight[FLOAT, 12x9x3x3]
  %linear.bias[FLOAT, 20]
  %linear2.bias[FLOAT, 15]
) {
  %9 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]](%input.1, %conv.weight, %conv.bias)
  %10 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], key = 'value', pads = [0, 0, 0, 0], strides = [1, 1]](%9, %conv2.weight, %conv2.bias)
  %11 = Relu(%10)
  %13 = MatMul[key = 'value'](%11, %18)
  %14 = Add(%13, %linear.bias)
  %16 = MatMul(%14, %19)
  %17 = Add(%16, %linear2.bias)
  return %17
}
```

`%10 = Conv` and `%13 = MatMul` has `key='value'` attribute but `%11 = Relu` hasn't. 
By using `apply_annotation` all node in the function is annotated.

```python
import pytorch_pfn_extras.onnx as tou

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Conv2d(6, 9, 3)
        self.conv2 = nn.Conv2d(9, 12, 3)
        self.linear = nn.Linear(28, 20)
        self.linear2 = nn.Linear(20, 15)

    def forward(self, x):
        h = self.conv(x)
        def _f(x):
            h = self.conv2(x)
            h = F.relu(h)
            h = self.linear(h)
            return h
        h = tou.apply_annotation(_f, h, key='value')
        h = self.linear2(h)
        return h

model = Net()
x = torch.randn((1, 6, 32, 32))
tou.export_testcase(model, x, '/path/to/outout')
onnx_proto = onnx.load(os.path.join('/path/to/output', 'model.onnx'))
print(onnx.helper.printable_graph(onnx_proto.graph))
```

```
graph torch-jit-export (
  %input.1[FLOAT, 1x6x32x32]
) initializers (
  %18[FLOAT, 28x20]
  %19[FLOAT, 20x15]
  %conv.bias[FLOAT, 9]
  %conv.weight[FLOAT, 9x6x3x3]
  %conv2.bias[FLOAT, 12]
  %conv2.weight[FLOAT, 12x9x3x3]
  %linear.bias[FLOAT, 20]
  %linear2.bias[FLOAT, 15]
) {
  %9 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]](%input.1, %conv.weight, %conv.bias)
  %10 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], key = 'value', pads = [0, 0, 0, 0], strides = [1, 1]](%9, %conv2.weight, %conv2.bias)
  %11 = Relu[key = 'value'](%10)
  %13 = MatMul[key = 'value'](%11, %18)
  %14 = Add(%13, %linear.bias)
  %16 = MatMul(%14, %19)
  %17 = Add(%16, %linear2.bias)
  return %17
}
```

Now `%11 = Relu` is annotated with `key='value'` attribute too.

### `scoped_anchor`

This annotates scope's beginning and end of one or modules by adding Anchor node.
Node would be named `Anchor_N_start` or `Anchor_N_end` (N is a index) and with op_type Identity.

* Adding custom parameter would add ONNX attribute and this will generate invalid ONNX in checker.
* Use this with `pytorch_pfn_extras.onnx.export_testcase` or `pytorch_pfn_extras.onnx.export`.
* When scope has multiple input/output only first input/output will get Anchor node added.
* `N` of node name is the index of pair beginning/end Anchor node like `Anchor_0_start`, `Anchor_0_end`.

```python
import pytorch_pfn_extras.onnx as tou

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Conv2d(6, 9, 3)
        self.conv2 = nn.Conv2d(9, 12, 3)
        self.linear = nn.Linear(28, 20)
        self.linear2 = nn.Linear(20, 15)

    def forward(self, x):
        h = self.conv(x)
        with tou.scoped_anchor(key='value'):
            h = self.conv2(h)
            h = self.linear(h)
        h = self.linear2(h)
        return h

    def forward(self, x):
        with annotate(key='value'):
            return self.add(x)

model = Net()
x = torch.randn((1, 6, 32, 32))
out_dir = tou.export_testcase(model, x, '/path/to/output')
onnx_proto = onnx.load(os.path.join('/path/to/output', 'model.onnx'))
print(onnx.helper.printable_graph(onnx_proto.graph))
```

```
graph torch-jit-export (
  %input.1[FLOAT, 1x6x32x32]
) initializers (
  %23[FLOAT, 28x20]
  %24[FLOAT, 20x15]
  %conv.bias[FLOAT, 9]
  %conv.weight[FLOAT, 9x6x3x3]
  %conv2.bias[FLOAT, 12]
  %conv2.weight[FLOAT, 12x9x3x3]
  %linear.bias[FLOAT, 20]
  %linear2.bias[FLOAT, 15]
) {
  %9 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]](%input.1, %conv.weight, %conv.bias)
  %11 = Identity[key = 'value'](%9)
  %12 = Conv[dilations = [1, 1], group = 1, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]](%11, %conv2.weight, %conv2.bias)
  %16 = MatMul(%12, %23)
  %17 = Add(%16, %linear.bias)
  %19 = Identity[key = 'value'](%17)
  %21 = MatMul(%19, %24)
  %22 = Add(%21, %linear2.bias)
  return %22
}
```

`%11 = Identity` (node name = `Anchor_0_start`) and `%19 = Identity` (node name = `Anchor_0_end`) is added. `key='value'` is added as ONNX attribute.

#### non-`nn.Module`

The target of scope is only `nn.Module`. You can add adding sub `nn.Module` instead, if scope bound doesn't match `nn.Module`.

```python
import pytorch_pfn_extras.onnx as tou

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        class _Net(nn.Module):
            def forward(self, x):
                return x + torch.ones((1,))
        self.add = _Net()

    def forward(self, x):
        with tou.scoped_anchor(key='value'):
            return self.add(x)

model = Net()
x = torch.randn((1, 6, 32, 32))
out_dir = tou.export_testcase(model, x, '/path/to/output')
onnx_proto = onnx.load(os.path.join('/path/to/output', 'model.onnx'))
print(onnx.helper.printable_graph(onnx_proto.graph))
``` 

```
graph torch-jit-export (
  %x.1[FLOAT, 1x6x32x32]
) {
  %2 = Identity[key = 'value'](%x.1)
  %3 = Constant[value = <Tensor>]()
  %4 = Add(%2, %3)
  %6 = Identity[key = 'value'](%4)
  return %6
}
```

Or you can use `anchor` (described below) instead.

### `anchor` (Future work)

Inserts Anchor node per each arbitrarily position of  `nn.Module` . Node name would be Anchor and op_type would be  `Identity`.

* Note: adding extra parameter would make extended ONNX format because it would be attribute.
* Please use it with `pytorch_pfn_extras.onnx.export_testcase` or `pytorch_pfn_extras.onnx.export`.
