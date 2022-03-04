CodeBlocks for Abstracting Logic Steps
========================================

The :class:`ppe.handler.CodeBlock <pytorch_pfn_extras.handler.CodeBlock>` API
provides a mean of abstracting the actions that are possible to be done in a model
in a device agnostic way.

Currently there is support for two different actions using ``CodeBlock``.

- :function:`ppe.handler.update_parameters <pytorch_pfn_extras.handler.update_parameters>`
 takes a model, an optimizer and returns a ``CodeBlock`` object that performs the forward, backward and optimizer step at once.

- :function:`ppe.handler.forward <pytorch_pfn_extras.handler.forward>`
 takes a model and returns a ``CodeBlock`` object that performs only the forward pass.

Executing CodeBlocks
-------------------------------

For executing ``CodeBlock`` objects we need to add an :method:`ppe.runtime.BaseRuntime.execute <pytorch_pfn_extras.runtime.BaseRuntime.execute` to the
corresponding ``Runtime`` class. This method takes a ``CodeBlock`` and uses the information in the object to execute the ``CodeBlock`` in the
device. Note that the :method:`ppe.runtime.PyTorchRuntime.execute <pytorch_pfn_extras.runtime.PyTorchRuntime.execute` method providesn support
for using PyTorch AMP with autocast or gradient scaling if needed.

Moreover, you can execute ``CodeBlock`` objects outside the training API.

.. code:: py

    ppe.to(model, "cuda:0")
    cblock = ppe.handler.update_parameters(model, optimizer)
    outs = cblock(input_batch)

The only requirement is that the associated model has been asigned a device using ``ppe.to``.
