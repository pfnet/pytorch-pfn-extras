from pytorch_extensions import ExtensionsManager
from pytorch_extensions import reporter
import pytorch_extensions.extensions as extensions

import time

max_epoch = 10
epoch_size = 938

# manager.extend(...) also works
my_extensions = [extensions.LogReport(),
                 extensions.ProgressBar(),
                 extensions.PrintReport(['epoch', 'iteration', 'loss'])]

models = {}
manager = ExtensionsManager(models, max_epoch, my_extensions)

current_it = 0
for epoch in range(max_epoch):
    #print(epoch)
    for iter in range(epoch_size):
        # Needs the total iters as in chainer
        current_it = epoch*epoch_size+iter
        with manager.run_iteration(
                epoch=epoch, iteration=current_it, epoch_size=epoch_size):
            reporter.report({'loss': iter/100+epoch})
            time.sleep(0.001)
