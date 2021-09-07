Trainer and Evaluator
============================

.. note::

   The Trainer/Evaluator APIs are currently under the technical preview and may subject to change in the future versions.

The Trainer and Evaluator provides the device-agnostic training framework for PyTorch.
These APIs abstract the training process using different :doc:`runtimes <runtimes>`, handlers, and :doc:`logics <logic>`.

Concepts
----------------------------------------

* **Trainer** (:func:`ppe.engine.create_trainer() <pytorch_pfn_extras.engine.create_trainer>`) abstracts the training loop, built on top of the :class:`ExtensionsManager <pytorch_pfn_extras.training.ExtensionsManager>`.
* **Evaluator** (:func:`ppe.engine.create_evaluator() <pytorch_pfn_extras.engine.create_evaluator>`) abstracts the evaluation step and invoked from the Trainer (usually once in every epoch).
* **Runtime** (:class:`ppe.runtime.BaseRuntime <pytorch_pfn_extras.runtime.BaseRuntime>`) represents an environment used to execute models. Device-specific implementations will reside here. PPE provides the default Runtime that supports the PyTorch-native devices (:class:`ppe.runtime.PyTorchRuntime <pytorch_pfn_extras.runtime.PyTorchRuntime>`).
* **Handler** (:class:`ppe.handler.Handler <pytorch_pfn_extras.handler.Handler>`) is a layer to support device-agnostic training. This is considered as a low-level API and in most cases users can just use the Handler provided by PPE.
* **Logic** (:class:`ppe.handler.Logic <pytorch_pfn_extras.handler.Logic>`) is a set of callback functions that define the training logic (`optimizer.zero_grad()`, forward, backward, `optimizer.step()`). You can inherit the class and define your own training flow in case you need more complex training processes such as GAN.
* **Model** is a :class:`torch.nn.Module` used for training and evaluation, whose inputs are dicts or keyword arguments and outputs of the ``forward`` pass is a dict.

Note that the default logic will perform ``backward`` in tensors returned by ``model.forward``
so you will need to perform the loss calculation inside the model itself.

Trainer at a glance
--------------------------

.. code:: python

    import torch
    import torch.nn.functional as F

    import pytorch_pfn_extras as ppe


    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.LazyLinear(1)

        def forward(self, *, x, target):
            y = self.w(x)
            loss = F.nll_loss(y, target)
            prefix = 'train' if self.training else 'val'
            ppe.reporting.report({f'{prefix}/loss': loss.item()})
            return {'loss': loss}


    model = MyModel()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)

    extensions = [
        ppe.training.extensions.LogReport(),
        ppe.training.extensions.ProgressBar(),
        ppe.training.extensions.PrintReport(
            ['epoch', 'iteration', 'train/loss', 'val/loss']),
    ]

    device = 'cuda:0'  # or any other PyTorch devices ('cpu', etc.) or PPE runtime names
    epochs = 10
    trainer = ppe.engine.create_trainer(
        model,
        optim,
        epochs,
        evaluator=ppe.engine.create_evaluator(
            model,
            device=device,
            progress_bar=True,
        ),
        device=device,
        extensions=extensions,
    )

    # Move the model to the device. This is almost equivalent to
    # `model.to(device)`, but supports PPE runtimes as well as the PyTorch's
    # built-in devices.
    ppe.to(model, device)

    # Using dummy data to illustrate the minimal working example.
    # Notice that dict keys match with the kwargs of the forward method.
    train_loader = torch.utils.data.DataLoader(
        [{'x': torch.rand(10, 64), 'target': torch.tensor([1])} for _ in range(1)],
        num_workers=8)
    val_loader = torch.utils.data.DataLoader(
        [{'x': torch.rand(10, 64), 'target': torch.tensor([1])} for _ in range(1)],
        num_workers=8)

    trainer.run(train_loader, val_loader)


Snapshot
----------------------------------------

To obtain and save the trained model for later use you can use the `Snapshot`
extension, or directly invoke `state_dict` on the trainer itself.

Handler
----------------------------------------

The ``ppe.handler.Handler`` object is used to help the trainer and evaluator objects
in the :doc:`Logic <logic>` and :doc:`Runtime <runtimes>` manipulation. This class
should ideally never be overriden by the user if the desired functionality can be
achieved through subclassing BaseLogic or BaseRuntime.

The handler object's main responsibility is to inspect all the submodules of a module
to obtain the runtimes they have associated, and then execute their callbacks
accordingly. In addition, it drives the actual model execution by using the user provided
Logic object and deals with asynchronous execution in runtimes that provide
support for it.

Runtime
------------------------

By inheriting :class:`ppe.runtime.BaseRuntime <pytorch_pfn_extras.runtime.BaseRuntime>` and implementing your own runtime, you can use your non-standard devices with the training loop.

.. code:: py

    class MyRuntime(BaseRuntime):
        ...

    # Register MyRuntime with device name "mydev"
    ppe.runtime.runtime_registry.register('mydev', MyRuntime)

    ppe.to(module_or_tensor, 'mydev')

See :doc:`runtimes` if you are interested in implementing your own runtime.
