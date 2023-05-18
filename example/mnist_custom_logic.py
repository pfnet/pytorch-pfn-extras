import argparse

import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as extensions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CustomLogic(ppe.handler.Logic):
    def __init__(self, steps_per_update):
        self.steps_per_update = steps_per_update
        super().__init__()

    def train_step(self, models, optimizers, batch_idx, batch):
        # Calls `optimizer.zero_grad()` at batch_idx % steps_per_update == 0.

        model = models[self.model_name]

        if batch_idx % self.steps_per_update == 0:
            optimizer = optimizers[self.model_name]
            optimizer.zero_grad()

        outs = model(**batch)
        for tensor in outs.values():
            tensor.backward()
        return outs

    def train_step_optimizers(self, models, optimizers, batch_idx):
        # Calls `optimizer.step()` at (batch_idx + 1) % steps_per_update == 0.

        if (batch_idx + 1) % self.steps_per_update == 0:
            optimizer = optimizers[self.model_name]
            optimizer.step()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="PyTorch device specifier"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--snapshot", type=str, default=None, help="path to snapshot file"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_cuda = args.device.startswith("cuda")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ppe.dataloaders.utils.CollateAsDict(
            names=["data", "target"]
        ),
        **kwargs,
    )  # type: ignore[arg-type]
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        collate_fn=ppe.dataloaders.utils.CollateAsDict(
            names=["data", "target"]
        ),
        **kwargs,
    )  # type: ignore[arg-type]

    model = Net()

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum
    )

    my_extensions = [
        extensions.LogReport(),
        extensions.ProgressBar(),
        extensions.observe_lr(optimizer=optimizer),
        extensions.ParameterStatistics(model, prefix="model"),
        extensions.VariableStatisticsPlot(model),
        extensions.PlotReport(
            ["train/loss", "val/loss"], "epoch", filename="loss.png"
        ),
        extensions.PrintReport(
            [
                "epoch",
                "iteration",
                "train/loss",
                "lr",
                "model/fc2.bias/grad/min",
                "val/loss",
                "val/accuracy",
            ]
        ),
        extensions.snapshot(),
    ]

    # Custom stop triggers can be added to the manager and
    # their status accessed through `manager.stop_trigger`
    trigger = None
    # trigger = ppe.training.triggers.EarlyStoppingTrigger(
    #     check_trigger=(1, 'epoch'), monitor='val/loss')

    class ModelWithLoss(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, data, target):
            output = model(data)

            if model.training:
                loss = F.nll_loss(output, target)
                ppe.reporting.report({"train/loss": loss.item()})
                return {"loss": loss}

            # Final result will be average of averages of the same size
            test_loss = F.nll_loss(output, target, reduction="mean").item()
            pred = output.argmax(dim=1, keepdim=True)
            return {"loss": test_loss, "output": pred}

    model_with_loss = ModelWithLoss(model)
    trainer = ppe.engine.create_trainer(
        model_with_loss,
        optimizer,
        args.epochs,
        device=args.device,
        extensions=my_extensions,
        stop_trigger=trigger,
        evaluator=ppe.engine.create_evaluator(
            model_with_loss,
            device=args.device,
            progress_bar=True,
            metrics=[ppe.training.metrics.AccuracyMetric("target", "output")],
            options={"eval_report_keys": ["loss", "accuracy"]},
        ),
        options={"train_report_keys": ["loss"]},
        logic=CustomLogic(3),
    )

    if use_cuda:
        ppe.to(model_with_loss, args.device)

    # Lets load the snapshot
    if args.snapshot is not None:
        state = torch.load(args.snapshot)
        trainer.load_state_dict(state)

    trainer.run(train_loader, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
