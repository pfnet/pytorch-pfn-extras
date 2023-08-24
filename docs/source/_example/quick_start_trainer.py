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

device = (
    "cuda:0"  # or any other PyTorch devices ('cpu', etc.) or PPE runtime names
)
epochs = 3
# Create a trainer with the defined model, optimizer, and other parameters
trainer = ppe.engine.create_trainer(
    models=model,
    optimizers=optimizer,
    max_epochs=epochs,
    evaluator=ppe.engine.create_evaluator(
        models=model,
        device=device,
    ),
    device=device,
)

# Send the model to device(GPU) for computation
ppe.to(model, device=device)

batch_size = 10
# Create 10 batches of random training data with dimension (batch_size x 64)
training_data = [
    {
        "x": torch.rand((batch_size, 64)),
        "target": torch.ones((batch_size,), dtype=torch.long),
    }
    for _ in range(10)
]
# Create 10 batches of random validation data with dimension (batch_size x 64)
validation_data = [
    {
        "x": torch.rand((batch_size, 64)),
        "target": torch.ones((batch_size,), dtype=torch.long),
    }
    for _ in range(10)
]

# Start the training and validation of the model
trainer.run(train_loader=training_data, val_loader=validation_data)

print("Finish training!")
