from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import pytorch_extensions as pte
import pytorch_extensions.extensions as extensions

# Extensions manager object
manager = None


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.1)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x, t):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.1)
        x = self.fc2(x)
        # pred_y = F.log_softmax(x, dim=-1)
        loss = F.cross_entropy(x, t)
        # if self.training:
        pte.reporter.report({
            'loss': loss.item(), 'acc': calc_accuracy(x, t)
        }, self)
        return loss

def calc_accuracy(y, t):
    acc = float((torch.argmax(y, dim=1) == t).sum()) / t.nelement()
    return acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cuda")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Set up a trainer
    optimizer.target = model
    # Updater needs to know the device to move the data to it.
    updater = pte.updaters.StandardUpdater(
        train_loader, optimizer, device=device)

    # manager.extend(...) also works
    my_extensions = [extensions.LogReport(),
                     extensions.ProgressBar(),
                     extensions.ExponentialShift('lr', 0.9999, optimizer, init=0.2, target=0.1),
                     extensions.observe_lr(optimizer=optimizer),
                     extensions.ParameterStatistics(model, prefix='model'),
                     extensions.VariableStatisticsPlot(model),
                     extensions.Evaluator(test_loader, model, progress_bar=True, device=device),
                     extensions.PlotReport(
                         ['main/loss', 'validation/main/loss'], 'epoch', filename='loss.png'),
                     extensions.PrintReport(['epoch', 'iteration',
                                             'main/loss', 'lr', 'model/fc2.bias/grad/min',
                                             'validation/main/loss', 'validation/main/acc'])]
    trainer = pte.Trainer(updater, (args.epochs, 'epoch'), extensions=my_extensions)
    trainer.run()

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
