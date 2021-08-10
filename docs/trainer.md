# Trainer Evaluator and Inferer

The trainer, Evaluator are interfaces to abstract the training
process using different [runtimes](runtimes.md) and [logics](logic.md).

## How to use

All these interfaces share a common point, inputs to the modules must be dicts
or kwargs, and outputs from the forward pass must be dicts too.

### Creating a Trainer and an Evaluator

In order to train a model, we provide a trainer like object that is able
to perform the train steps and evaluation using a logic that performs a 
simple forward, backward and optimizer step loop.
All the training steps are abstracted through the use of the [runtimes](runtimes.md)
interface, allowing the user to perform trainer in multiple devices, or customize a handler
for more complex training processes such as GANs by using the [logic](logic.md) interface.

The trainer can be created through the `ppe.engine.create_trainer` function and it needs
the model, optimizer and number of epochs as an argument. Also the logic to
use during the training and evaluation steps, the device or the handle
where the user wants to perform the train and the list of extensions to run.

To obtain and save the trained model for later use you can use the `Snapshot`
extension, or directly invoke `state_dict` on the trainer itself.

To perform evaluation in the model at the end of every training epoch, an Evaluator object 
can be created using `ppe.engine.create_evaluator`.
A reference to the model and the device to run the validation is only needed.

Note that the default logic  will perform `backward` in tensors returned by `model.forward`
so you will need to perform the loss calculation inside the model itself.


```python
import torch
import torch.nn.functional as F
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions

import dataset


class MyModel(torch.nn.Module):
    def __init__(self):
        self.w = torch.nn.Linear(10, 1)

    def forward(self, *, x):
        y = self.w(x)
        loss = F.nll_loss(output, target)
        prefix = 'train' if self.training else  'val'
        ppe.reporting.report({f'{prefix}/loss': loss.item()})
        return loss


model = MyModel()
optim = torch.optim.SGD(model.parameters(), lr=0.01)

extensions = [ppe.training.extensions.LogReport(),
              ppe.training.extensions.ProgressBar(),
              ppe.training.extensions.PrintReport(['epoch', 'iteration',
                                'train/loss', 'val/loss']),

epochs = 10
trainer = ppe.engine.create_trainer(
    model,
    optim,
    epochs,
    evaluator=ppe.engine.create_evaluator(
        model,
        device='cuda:0',
        progress_bar=True,
    ),
    device='cuda:0',
    extensions=extensions)

# Any iterable is fine
train_loader = torch.utils.data.DataLoader(dataset.TrainDataset(), num_workers=8)
val_loader = torch.utils.data.DataLoader(dataset.ValDataset(), num_workers=8)

trainer.run(train_loader, val_loader)
```

### Trainer Handler

The ppe.handler.Handler object is used to help the trainer and evaluator objects
in the [logic](logic.md) and [runtime](runtimes.md) manipulation. This class
should ideally never be overriden by the user if the desired functionality can be
achieved through subclassing BaseLogic or BaseRuntime.

The Handler object main responsibility is to inspect all the submodules of a module
to obtain the runtimes they have associated, and then execute their callbacks
accordingly. In addition, it drives the actual model execution by using the user provided
Logic object and deals with asynchronous execution in runtimes that provide
support for it.
