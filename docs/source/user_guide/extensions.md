# Extensions Manager

Extensions Manager provides an interface to extend your training loop, by integrating it into your manual training loop or Ignite.

## Extensions

See :doc:`../reference/index` for the list of built-in extensions.

## How to use

Create an `ExtensionsManager` object and then wrap the iteration of your
training loop inside the `manager.run_iteration()` context manager.

An example follows:

```python
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions

import time
import math

max_epoch = 10
iters_per_epoch = 938

# manager.extend(...) also works
my_extensions = [extensions.LogReport(),
                 extensions.ProgressBar(),
                 extensions.PrintReport(['epoch', 'iteration', 'sin', 'cos'])]

models = {}
optimizers = []
manager = ppe.training.ExtensionsManager(
    models, optimizers, max_epoch,
    extensions=my_extensions,
    iters_per_epoch=iters_per_epoch)

for epoch in range(max_epoch):
    for i in range(iters_per_epoch):
        with manager.run_iteration():
            ppe.reporting.report({
                'sin': math.sin(i * 2 * math.pi / iters_per_epoch),
                'cos': math.cos(i * 2 * math.pi / iters_per_epoch),
            })
            time.sleep(0.001)
```

In the examples folder there is a mnist using all the avaiable extensions.

### Usage with Ignite

Ignite is supported by using the `IgniteExtensionsManager` with the trainer
as the first argument.

The user needs to define an ignite event to report the appropiated metrics
for the extensions to use them.


```python
manager = ppe.training.IgniteExtensionsManager(
    trainer, models, optimizers, epochs,
    extensions=my_extensions)

@trainer.on(Events.ITERATION_COMPLETED)
def report_loss(engine):
    ppe.reporting.report({'train/loss':engine.state.output})
```


### Using Evaluators

#### Regular PyTorch

In order to report the results of the evaluation so they can be
accessed by other extensions, an `Evaluation` extension
needs to be created with the argument `eval_func` set to a function
that gets the current data and target batches as parameters and
reports the needed metrics. [Example](https://github.com/pfnet/pytorch-pfn-extras/blob/master/example/mnist.py#L49-L64)

The test function looks has the following signature
```python
def test(args, model, device, data, target):
```
and is invoked once per batch in the validation dataloader.
It is important to report the current validation loss or accuracy in order to the log report to see it.

```python
def test(args, model, device, data, target):
    ...
    # Final result will be average of averages of the same size
    test_loss += F.nll_loss(output, target, reduction='mean').item()
    ppe.reporting.report({'val/loss': test_loss})
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    ppe.reporting.report({'val/acc': correct/len(data)})
```

#### Ignite

Just use the `IgniteEvaluator` extension with the ignite created evaluator as
the first parameter and you are ready to go. [Example](https://github.com/pfnet/pytorch-pfn-extras/blob/master/example/ignite-mnist.py#L79-L80)
The metrics defined when creating the evaluator with `create_supervised_evaluator` will be automatically reported
```python
 create_supervised_evaluator(model, metrics={'acc': Accuracy(), 'loss': Loss(F.nll_loss)}, device)
```

### Snapshots

It is possible to take snapshots by using the [`snapshot`](https://github.com/pfnet/pytorch-pfn-extras/blob/master/example/mnist.py#L142)
training extension just as in chainer.

Whenever the extension is triggered, it saves the status of the optimizer, model and extensions to the output folder in the same way as chainer.
To load the snapshot and continue the training call `torch.load` and use the `ExtensionsManager.load_state_dict`[example](https://github.com/pfnet/pytorch-pfn-extras/blob/master/example/mnist.py#L155-L157) to resume the training.
The snapshots can be used outside the pytorch-pfn-extras module just by accessing the models, or optimizers fields of the loaded state.

### Extensions execution order

The supported extensions honours the chainer priorities for execution.
However, when using Ignite. Chainer extensions are executed after any user-defined ignite events.
The idea is to use ignite events to report the metrics of the model, and after this, Chainer extensions will be
executed in the chainer defined order.

If you want to execute an event-handler in between chainer extensions, create a Chainer-like extension
and access the ignite engine on the `.engine` attribute of the manager object passed as a parameter
when your extension is called.

# Creating Extensions

It is possible to create an extension just by passing a function which
receives the manager object as an argument to the manager extend call

```python
def my_extension(manager):
    print('Epoch-Iteration: {}-{}'.format(manager.epoch, manager.iteration)

manager.extend(my_extension, trigger=(1, 'iteration')
```

It is also possible to create extensions using the `ppe.training.extension.make_extension`
decorator to add a specific `trigger`, `default_name`, `priority`.
In addition, `initializer`, `finalizer` and `on_error` functions can be specified as well.

```python
@ppe.training.extension.make_extension(finalizer=lambda: print('done'))
def my_extension(manager):
    print('Epoch-Iteration: {}-{}'.format(manager.epoch, manager.iteration)
```

Finally, it is possible to create an extension by subclassing the `ppe.training.extensions.Extension` class
as shown below.

```python
import pytorch_pfn_extras as ppe

class MyExtension(ppe.training.extension.Extension)
    def __init__(self, args):
        self.args = args

    def initialize(self, manager):
        """
        Automatically called before training. Optional.
        """
        pass

    def __call__(self, manager):
        """
        Called when the associated trigger is fired.
        """
        print('Epoch-Iteration: {}-{}'.format(manager.epoch, manager.iteration)

    def state_dict(self):
        """ 
        Used to serialize the state. Optional.
        """
        return {'args': self.args}

    def load_state_dict(self, state):
        """ 
        Used to deserialize the state. Optional.
        """
        self.args = state['args']
```
