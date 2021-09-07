Logic for Custom Training and Evaluation
============================================================

In the training and evaluation engines, :class:`ppe.handler.BaseLogic <pytorch_pfn_extras.handler.BaseLogic>` API is in charge of abstracting the algorithmic details of the training and evaluation loops.

Logic is an object that defines multiple callbacks used
through the training and evaluation processes.
With logic, we can implement training of complex models such as GANs.

Users wanting to define their own Logic for training can inherit from
:class:`ppe.handler.Logic <pytorch_pfn_extras.handler.Logic>` which implements the training and evaluation steps to train
a single module.

Logic functions are not exepcted to be directly called by the user.
They will be invoked by the Trainer and Evaluator engines.

Default Logic (:class:`ppe.handler.Logic <pytorch_pfn_extras.handler.Logic>`)
------------------------------------------------------------------------------------------

PPE provides a default logic that performs the forward/backward/optimizer loop
for a single model. This logic allows using some torch features such as AMP
and GradScaler and performs the backward pass on the outputs specified by the
config option backward_outputs.
