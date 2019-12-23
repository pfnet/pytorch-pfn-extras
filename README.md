# pytorch-extensions

Native port to pytorch of Chainer extensions.

No chainer or chainer-pytorch-interop required.

# What is supported

Chainer training engine with extensions reporter and triggers is fully supported

Currently working extensions

+ Evaluator
+ ExponentialShift
+ LogReport
+ MicroAverage
+ PrintReport
+ ProgressBar
+ ParameterStatistics
+ PlotReport
+ observe_lr
+ observe_value
+ snapshot
+ VariableStatisticsPlot

# How to use with `Trainer`

[Example](https://github.pfidev.jp/ecastill/pytorch-extensions/blob/master/example/trainer-mnist.py#L87-L111)

```python
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
# Set up a trainer
optimizer.target = model
# Updater needs to know the device to move the data to it.
# already knows
updater = pte.updaters.StandardUpdater(
    train_loader, optimizer, device=device)
trainer = pte.Trainer(updater, (args.epochs, 'epoch'), extensions=my_extensions)
trainer.run()
```
One of the differences with Chainer is the need to transfer the model to the device before creating the
`optimizer`. In PyTorch, the `Optimizer` class needs to have the associated model parameters in the device
memory at creation time. This prevents the `trainer` and `updater` to move the model to the corresponding device.

The default convert function has been changed to do device transferences only, since PyTorch `DataLoader` class
already returns the data in the desired format.

Support for snapshots is on-going.

# How to use without `Trainer`

Since there is no trainer object in regular Pytorch, you have to create a
`ExtensionsManager` object and then wrap the iteration of your training loop inside the
`manager.run_iteration()` context manager.

An example follows:

```python
from pytorch_extensions import ExtensionsManager
from pytorch_extensions import reporter
import pytorch_extensions.extensions as extensions

import time

max_epoch = 10
epoch_size = 938

# manager.extend(...) also works
my_extensions = [extensions.LogReport(),
                 extensions.ProgressBar(),
                 extensions.PrintReport(['epoch', 'iteration', 'loss'])]

models = {}
manager = ExtensionsManager(models, max_epoch, my_extensions)

current_it = 0
for epoch in range(max_epoch):
    #print(epoch)
    for iter in range(epoch_size):
        # Needs the total iters as in chainer
        current_it = epoch*epoch_size+iter
        with manager.run_iteration(
                epoch=epoch, iteration=current_it, epoch_size=epoch_size):
            reporter.report({'loss': iter/100+epoch})
            time.sleep(0.001)
```

In the examples folder there is a mnist using all the avaiable extensions.

## Ignite

Ignite is supported by using the `IgniteExtensionsManager` with the trainer
as the first argument.

The user needs to define a ignite event to report the appropiated metrics
for the extensions to use them.


```python
@trainer.on(Events.ITERATION_COMPLETED)
def report_loss(engine):
    pte.reporter.report({'train/loss':engine.state.output})
```


# Using Evaluators

## Regular PyTorch

In order to report the results of the evaluation so they can be
accessed by other extensions, an `Evaluation` extension
needs to be created with the argument `eval_func` set to a function
that gets the current data and target batches as parameters and
reports the needed metrics. [Example](https://github.pfidev.jp/ecastill/pytorch-extensions/blob/master/example/mnist.py#L51-L66)

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
    pte.reporter.report({'val/loss': test_loss})
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    pte.reporter.report({'val/acc': correct/len(data)})
```
## Ignite

Just use the `IgniteEvaluator` extension with the ignite created evaluator as
the first parameter and you are ready to go. [Example](https://github.pfidev.jp/ecastill/pytorch-extensions/blob/master/example/ignite-mnist.py#L73-L75)
The metrics defined when creating the evaluator with `create_supervised_evaluator` will be automatically reported
```python
 create_supervised_evaluator(model, metrics={'acc': Accuracy(), 'loss': Loss(F.nll_loss)}, device)
```

# Snapshots

It is possible to take snapshots by using the [`snapshot`](https://github.pfidev.jp/ecastill/pytorch-extensions/blob/1aa0fa47ad972d1514b034fdb05afcb3e7eef538/example/mnist.py#L133)
training extension just as in chainer.

Whenever the extension is triggered, it saves the status of the optimizer, model and extensions to the output folder in the same way as chainer.
To load the snapshot and continue the training call `torch.load` and use the `ExtensionsManager.load_state_dict`[example](https://github.pfidev.jp/ecastill/pytorch-extensions/blob/a5d1d356b7a53e793423f334137f8134edca089b/example/mnist.py#L139-L141) to resume the training.
The snapshots can be used outside the pytorch-extensions module just by accessing the models, or optimizers fields of the loaded state.

# Extensions execution order

The supported extensions honours the chainer priorities for execution.
However, when using Ignite. Chainer extensions are executed after any user-defined ignite events.
The idea is to use ignite events to report the metrics of the model, and after this, Chainer extensions will be
executed in the chainer defined order.

If you want to execute an event-handler in between chainer extensions, create a Chainer-like extension
and access the ignite engine on the `.engine` attribute of the manager object passed as a parameter
when your extension is called.
