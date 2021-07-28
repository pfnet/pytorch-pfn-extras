# Lazy Modules

Lazy modules can automatically infer shapes of parameters based on the shape of the data given to the first forward invocation.

Following modules are provided:

* `ppe.nn.LazyLinear`
    * Module that behaves as `torch.nn.Linear` but `in_features` can be set to `None`.
    * This module is now included as a part of PyTorch 1.8 release ([torch.nn.LazyLinear](https://pytorch.org/docs/stable/generated/torch.nn.LazyLinear.html), [pull-request](https://github.com/pytorch/pytorch/pull/44538)).

* `ppe.nn.LazyConv1d`, `ppe.nn.LazyConv2d`, `ppe.nn.LazyConv3d`
    * Module that behaves as `torch.nn.Conv[123]d` but `in_channels` can be set to `None`.
    * These modles are now included as a part of PyTorch 1.8 release ([torch.nn.LazyConvXd](https://pytorch.org/docs/stable/generated/torch.nn.LazyConv1d.html), [pull-request](https://github.com/pytorch/pytorch/pull/47350)).

* `ppe.nn.LazyBatchNorm1d`, `ppe.nn.LazyBatchNorm2d`, `ppe.nn.LazyBatchNorm3d`
    * Module that behaves as `torch.nn.BatchNorm[123]d` but `num_features` can be set to `None`.
    * These modles are now included as a part of PyTorch 1.9 release ([torch.nn.LazyBatchNormXd](https://pytorch.org/docs/stable/generated/torch.nn.LazyBatchNorm1d.html), [pull-request](https://github.com/pytorch/pytorch/pull/51862)).

Now that all lazy modules are merged to the upstream, we encourage you to migrate to PyTorch's lazy modules.
We will keep these implementaions only for backward compatibility.

Note that you need to run a "dummy" forward to initialize lazy parameters.
See the example below:

```py
import torch
import torch.nn.functional as F

import pytorch_pfn_extras as ppe


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ppe.nn.LazyConv2d(None, 20, 5, 1)
        self.conv2 = ppe.nn.LazyConv2d(None, 50, 5, 1)
        self.fc1 = ppe.nn.LazyLinear(None, 500)
        self.fc2 = ppe.nn.LazyLinear(None, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()

# Initialize lazy parameters.
dummy_input = ...
model(dummy_input)

# Pass parameters to the optimizer.
optimizer = torch.optim.SGD(
    model.parameters(), lr=args.lr, momentum=args.momentum)

# Run training loop.
# ...
```

You need to run a dummy forward before passing parameters to optimizers; otherwise optimizers cannot refer to lazily-initialized parameters.
You will get a warning if you pass uninitialized lazy parameters to optimizers:

```
>>> model = ppe.nn.LazyLinear(None, 10)
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
/.../pytorch-pfn-extras/pytorch_pfn_extras/nn/modules/lazy.py:127: UserWarning:
    Use of uninitialized lazy parameter in Optimizer has been detected.
    Maybe you forgot to run forward before passing `module.parameters()` to the optimizer?
```
