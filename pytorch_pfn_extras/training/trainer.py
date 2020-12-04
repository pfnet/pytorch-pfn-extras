import torch

import pytorch_pfn_extras as ppe


class Trainer(ppe.engine._Engine):
    def __init__(
        self,
        backend,
        models,
        optimizers,
        max_epochs,
        *,
        run_fn=None,
        iters_per_epoch,
        evaluator=None,
        extensions=None,
        out_dir="result",
        stop_trigger=None,
        to_report_outputs=[],
        writer=None,
        device_options=None,
    ):
        super().__init__(
            models,
            optimizers,
            max_epochs,
            iters_per_epoch=iters_per_epoch,
            extensions=extensions,
            out_dir=out_dir,
            stop_trigger=stop_trigger,
            device_options=device_options,
        )
        self._backend = backend
        self._run_fn = run_fn
        self.evaluator = evaluator
        self.val_loader = None
        self.to_report_outputs = to_report_outputs

    @property
    def epoch(self):
        return self.updater.epoch

    @property
    def epoch_detail(self):
        return self.updater.epoch_detail

    @property
    def iteration(self):
        return self.updater.iteration

    @property
    def is_before_training(self):
        return self.updater.iteration == 0

    @property
    def stop_trigger(self):
        return self._stop_trigger

    @stop_trigger.setter
    def stop_trigger(self, trigger):
        self._stop_trigger = trigger

    def get_optimizer(self, name):
        return self._manager.optimizers[name]

    def set_optimizer(self, name, optimizer):
        self._manager.optimizers[name] = optimizer

    def is_epoch_last_iter(self, idx):
        return (idx + 1) == (self._manager._iters_per_epoch - 1)

    def takes_ckpt_this_iter(self, idx):
        return False

    def get_to_report_outputs(self):
        return self.to_report_outputs

    def run(self, train_loader, val_loader=None):

        self._backend.setup_train(self, train_loader, self._device_options)
        if self.evaluator is not None:
            self._backend.setup_evaluation(
                self.evaluator, val_loader, self._device_options
            )
        while not self._manager.stop_trigger:
            self._backend.pre_train_step(self)
            for idx, x in enumerate(train_loader):
                with self._manager.run_iteration(step_optimizers=["main"]):
                    if self._run_fn is None:
                        outs = self._backend.train_step(self, idx, x)
                    else:
                        outs = self._run_fn(self.models, x)
                    self._backend.process_train_step_outputs(self, outs)
                    if (
                        self.is_epoch_last_iter(idx)
                        and self.evaluator is not None
                    ):
                        self._backend.pre_validation(self, self.evaluator)
                        self.evaluator.run(val_loader)


def create_trainer(device, *args, **kwargs):
    # Get the backend
    backend = ppe.backend._backend_dispatcher.dispatch_backend(device)
    return Trainer(backend, *args, **kwargs)


if __name__ == "__main__":
    from pytorch_pfn_extras.training import extensions

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(10, 15)
            self.l2 = torch.nn.Linear(15, 20)

        def forward(self, x):
            y = self.l1(x)
            y = self.l2(y.to("cpu"))
            return y.sum()

    class MyModelWithLossFn(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.loss_fn = torch.nn.L1Loss()

        def forward(self, x, t):
            y = self.model(x)
            prefix = "train" if self.training else "val"
            loss = self.loss_fn(y, t)
            ppe.reporting.report({f"{prefix}/loss": loss})
            return loss

    model = MyModel()
    model_with_loss_fn = MyModelWithLossFn(model)
    input = torch.randn((10,), requires_grad=True)
    target = torch.randn((20,))
    loss = model_with_loss_fn(input, target)
    print(loss)
    loss.backward()
    print(input.grad)

    # Lets make the example trainable

    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    data = [(torch.rand(10,), torch.rand(20,),) for i in range(10)]
    evaluator = ppe.training.create_evaluator("cpu", model_with_loss_fn)

    trainer_extensions = [
        extensions.LogReport(trigger=(10, "iteration")),
        extensions.ProgressBar(update_interval=2),
        extensions.PrintReport(
            [
                "epoch",
                "iteration",
                "train/loss",
                "val/loss",
                "val/accuracy",
                "elapsed_time",
                "time",
            ]
        ),
        extensions.PlotReport(
            ["train/loss"], "epoch", filename="train_loss.png"
        ),
        extensions.PlotReport(["val/loss"], "epoch", filename="val_loss.png"),
        extensions.PlotReport(
            ["val/accuracy"], "epoch", filename="val_accuracy.png"
        ),
        extensions.PlotReport(["time"], "epoch", filename="time.png"),
        extensions.snapshot(),
    ]

    trainer = create_trainer(
        "cpu",
        model_with_loss_fn,
        optim,
        20,
        iters_per_epoch=10,
        evaluator=evaluator,
        extensions=trainer_extensions,
    )
    trainer.run(data, data)
