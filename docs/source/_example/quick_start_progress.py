import pytorch_pfn_extras as ppe
import torch


class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(in_features=64, out_features=2)
        self.criterion = torch.nn.NLLLoss()

    def forward(self, x, target):
        y = self.linear.forward(x).log_softmax(dim=1)
        loss = self.criterion.forward(y, target)
        return {"loss": loss}


model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

extensions = [
    ppe.training.extensions.LogReport(),
    ppe.training.extensions.ProgressBar(),  # Displaying the training progress.
    ppe.training.extensions.PrintReport(  # Displays the collected logs interactively.
        [
            "epoch",  # epoch, iteration, elapsed_time are automatically collected by LogReport.
            "iteration",
            "elapsed_time",
            "train/loss",  # The parameters specified by train_report_keys are collected under keys with the 'train/' prefix.
            "val/loss",  # The parameters specified by eval_report_keys are collected under keys with the 'val/' prefix.
        ],
    ),
]

device = "cuda:0"
epochs = 3
trainer = ppe.engine.create_trainer(
    models=model,
    optimizers=optimizer,
    max_epochs=epochs,
    extensions=extensions,
    evaluator=ppe.engine.create_evaluator(
        models=model,
        device=device,
        options={
            "eval_report_keys": ["loss"],
        },
    ),
    device=device,
    options={
        "train_report_keys": ["loss"],
    },
)

ppe.to(model, device=device)

batch_size = 10
training_data = [
    {
        "x": torch.rand((batch_size, 64)),
        "target": torch.ones((batch_size,), dtype=torch.long),
    }
    for _ in range(10)
]
validation_data = [
    {
        "x": torch.rand((batch_size, 64)),
        "target": torch.ones((batch_size,), dtype=torch.long),
    }
    for _ in range(10)
]

trainer.run(train_loader=training_data, val_loader=validation_data)

print("Finish training!")
