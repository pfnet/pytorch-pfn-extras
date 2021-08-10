# Logic for Custom Training and Evaluation

In the training and evaluation engines, [interface](trainer.md) 
the logic API is in charge of abstracting the algorithmic details 
of the training and evaluation loops.

Logic is an object that defines multiple callbacks used
through the training and evaluation processes.
With logic, we can implement training of complex models such as GANs.

The logic interface is as follows:

```python
class BaseLogic:

    def train_step(self, models, optimizers, batch_idx, batch):
        # Definition of the training step with access to all the modules
        # and optimizers.

    def train_step_optimizers(self, optimizers, batch_idx):
        # This is used to step the optimizers after the forward and backward
        # passes of train_step are complete.

    def eval_step(self, model, batch_idx, batch):
        # Definition of the evaluation step with access to all the models.
```

Users wanting to define their own Logic for training can inherit from
ppe.handler.Logic which implements the training and evaluation steps to train
a single module.

BaseLogic functions should never be directly called by the user.
They will be invoked by the Trainer and Evaluator engines by means of the
ppe.handler.Handler object.

## ppe.handler.Logic

PPE provides a default logic that performs the forward/backward/optimizer loop
in a single model. This logic allows using some torch features such as AMP
and GradScaler and performs the backward pass on the outputs specified by the
config option backward_outputs.
