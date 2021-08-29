# Runtimes for Custom Devices

The runtime API is in charge of abstracting the device details 
and performing the movement of data and modules to the corresponding device.

A runtime is an object that defines multiple callbacks used
through the training, evaluation, and regular model calls.
With runtimes, we can implement training in devices other than cpus or gpus
with minimal changes to the user code.

The runtime interface is as follows:

```python
class BaseRuntime:

    def __init__(self, device_spec, config=None):
        # Consumes the device configuration options
        ...

    # Functions related to data & module movement
    def convert_batch(self, args):
        # Transfers the given batch to the specific device.
        ...

    def move_module(self, module):
        # Transfers the module to the specific device.
        ...

    def move_tensor(self, tensor):
        # Transfers the tensor to the specific device.
        ...

    def initialize_module(self, module, loader_or_batch, optimizer=None):
        # Initializes the module at the beginning of training or inference.
        ...

    # Helper functions for the Trainer module
    def train_epoch_begin(self, module):
        # Preprocess of each epoch.
        ...

    def train_pre_step(self, trainer, module, batch_idx, batch):
        # Called at the beginning of every iteration.
        ...

    def train_post_step(self, trainer, module, batch_idx, batch, outputs):
        # Called at the end of every iteration.
        ...

    def train_validation_begin(self, module):
        # Called before the evaluation starts.
        ...

    # Helper functions for the Evaluator module
    def eval_pre_step(self, evaluator, module, batch_idx, batch):
        # Called before each evaluation step.
        ...

    def eval_post_step(self, evaluator, module, batch_idx, batch, outputs):
        # Called after each evaluation step.
        ...

    def get_pending_result(self, moduile, blocking):
        # Called to retrieve the result of an asynchronous call.
        ...
```

Users wanting to override only a few callbacks can inherit from
ppe.runtime.PyTorchRuntime which implements the basic functionality for cpu and
gpu devices.

Runtimes must be registered by calling the
ppe.runtime.runtime_registry.register(device_type, runtime_class) function
for them to be discoverable.


## Use of ppe.to to transfer modules and batches to custom devices

If you have defined a new runtime for a custom device the ppe.to function
allows moving a module or a tensor to the new device by invoking the]
Runtime.move_tensor and Runtime.move_module when needed.

The module (or submodule) will be tagged by adding a attribute named
`_ppe_runtime` that holds the needed runtime. It is the responsibility of the
user written runtime to perform the actual movement to the device and apply
all the transformations needed to a module so it can be correctly executed.

Usually, runtime writers will need to replace the given module forward function
by a new one that performs the actual device execution.

```python
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(x):
        return self.layer(x)

class MyMagicDeviceRuntime(ppe.runtime.BaseRuntime):
    def _device_forward(self, args):
        return run_batch_in_my_device(args):

    def move_module(self, module):
        # Registers a hook to initialize the module on the first batch
        # execution
        def hook(module, *args):
            module._ppe_runtime.initialize_module(module, args)

        self.hook = module.register_forward_pre_hook(hook)
        # Change the module forward to do the computation in the device
        module.forward = self._device_forward

    def initialize_module(self, module, loader_or_batch, optimizer=None):
        create_the_module_in_my_device(module, loader_or_batch, optimizer) 

# Register the runtime class
ppe.runtime.runtime_registry.register('my_device', MyMagicDeviceRuntime)

# Create a regular module
module = MyModule()
# Move the module to the device
ppe.to(module, device='my_device')

for x in my_dataloader:
    # The first iteration will create the module in the device
    # and the next ones will directly execute the module in the device instead
    # of executing the regular pytorch `forward` call.
    y = model(x)
```

Please note that this is an oversimplified description and that developing a
runtime that is 100% compatible with PyTorch requires to wrap the substitute
forward function with torch.autograd.Function among several other concerns
such as state_dict manipulation to ensure correcteness.
