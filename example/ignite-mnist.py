from argparse import ArgumentParser

from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST

from ignite.engine import Events
from ignite.engine import create_supervised_trainer
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as extensions


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def get_data_loaders(train_batch_size, val_batch_size):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_loader = DataLoader(
        MNIST(download=True, root="../data", transform=data_transform,
              train=True),
        batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(
        MNIST(download=False, root="../data", transform=data_transform,
              train=False),
        batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval):
    train_loader, val_loader = get_data_loaders(
        train_batch_size, val_batch_size)
    model = Net()
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    model = model.to(device)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer.step()
    trainer = create_supervised_trainer(
        model, optimizer, F.nll_loss, device=device)
    evaluator = create_supervised_evaluator(
        model,
        metrics={'acc': Accuracy(), 'loss': Loss(F.nll_loss)},
        device=device)

    # manager.extend(...) also works
    my_extensions = [
        extensions.LogReport(),
        extensions.ProgressBar(),
        extensions.observe_lr(optimizer=optimizer),
        extensions.ParameterStatistics(model, prefix='model'),
        extensions.VariableStatisticsPlot(model),
        extensions.snapshot(),
        extensions.IgniteEvaluator(
            evaluator, val_loader, model, progress_bar=True),
        extensions.PlotReport(['train/loss'], 'epoch', filename='loss.png'),
        extensions.PrintReport([
            'epoch', 'iteration', 'train/loss', 'lr',
            'model/fc2.bias/grad/min', 'val/loss', 'val/acc',
        ]),
    ]
    models = {'main': model}
    optimizers = {'main': optimizer}
    manager = ppe.training.IgniteExtensionsManager(
        trainer, models, optimizers, args.epochs,
        extensions=my_extensions)

    # Lets load the snapshot
    if args.snapshot is not None:
        state = torch.load(args.snapshot)
        manager.load_state_dict(state)

    @trainer.on(Events.ITERATION_COMPLETED)
    def report_loss(engine):
        ppe.reporting.report({'train/loss': engine.state.output})

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1000,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging '
                             'training status')
    parser.add_argument('--snapshot', type=str, default=None,
                        help='path to snapshot file')

    args = parser.parse_args()

    run(args.batch_size, args.val_batch_size, args.epochs, args.lr,
        args.momentum, args.log_interval)
