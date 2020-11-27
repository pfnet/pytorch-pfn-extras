import torch

import pytorch_pfn_extras as ppe


def test_lr_scheduler():
    param = torch.nn.Parameter(torch.zeros(10))
    optim = torch.optim.SGD([param], 1.0)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=[10, 20, 30], gamma=0.1, last_epoch=-1
    )
    ext = ppe.training.extensions.LRScheduler(sched)
    manager = ppe.training.ExtensionsManager(
        {}, {'main': optim}, 1, extensions=[ext], iters_per_epoch=40)
    for i in range(40):
        with manager.run_iteration(step_optimizers=['main']):
            if i < 10:
                assert optim.param_groups[0]['lr'] > 0.99
                assert optim.param_groups[0]['lr'] < 1.1
            elif i < 20:
                assert optim.param_groups[0]['lr'] > 0.09
                assert optim.param_groups[0]['lr'] < 0.11
            elif i < 30:
                assert optim.param_groups[0]['lr'] > 0.009
                assert optim.param_groups[0]['lr'] < 0.011
            elif i < 40:
                assert optim.param_groups[0]['lr'] > 0.0009
                assert optim.param_groups[0]['lr'] < 0.0011
