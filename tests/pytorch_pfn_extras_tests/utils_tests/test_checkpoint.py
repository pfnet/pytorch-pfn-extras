import pytest
import torch

import pytorch_pfn_extras as ppe


class SubNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 5, 3, 1, 1)
        self.bn1 = torch.nn.BatchNorm2d(5)

    def forward(self, x):
        return self.bn1(self.conv1(x)).relu()


class Net(torch.nn.Module):
    def __init__(self, checkpoint_type):
        super().__init__()
        self.checkpoint_type = checkpoint_type

        self.conv1 = torch.nn.Conv2d(1, 5, 3, 1, 1)
        self.bn1 = torch.nn.BatchNorm2d(5)
        self.part1 = SubNet()
        self.part2 = SubNet()

    def forward(self, x):
        x = self.bn1(self.conv1(x)).relu()

        if self.checkpoint_type == 'none':
            x = self.part1(x)
        elif self.checkpoint_type == 'bnaware':
            x = ppe.utils.checkpoint.checkpoint(self.part1, x)

        x = self.part2(x)

        return x


def _get_bn_stats_test_checkpoint(cp_type):
    torch.manual_seed(42)
    net = Net(cp_type)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    h, w = 32, 32

    bn = net.part1.bn1

    for _ in range(2):
        x = torch.arange(2 * h * w).reshape((2, 1, h, w)).float()

        opt.zero_grad()
        y = net(x)
        y.sum().backward()
        opt.step()

    return (bn.weight, bn.bias, bn.running_mean, bn.running_var)


@pytest.mark.gpu
def test_checkpoint():
    baseline = _get_bn_stats_test_checkpoint('none')
    ckpt = _get_bn_stats_test_checkpoint('bnaware')
    for p_b, p_c in zip(baseline, ckpt):
        assert torch.allclose(p_b, p_c)
