import warnings
import ignite
from pytorch_pfn_extras.training import Updater


class MixedFP16Updater(Updater):
    def __init__(self, model, optimizer, loss_fn,
                 device=None, non_blocking=False,
                 prepare_batch=ignite.engine._prepare_batch,
                 output_transform=lambda x, y, y_pred, loss: loss.item()):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.non_blocking = non_blocking
        self.output_transform = output_transform
        self.prepare_batch = prepare_batch

        if self.device:
            self.model.to(self.device)

        try:
            import apex
            self.apex_available = True
        except:
            warnings.warn("Need apex for MixedFP16 training but failed to import")
            self.apex_available = False

    def __call__(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()
        x, y = self.prepare_batch(batch, device=self.device, non_blocking=self.non_blocking)
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        if self.apex_available:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self.optimizer.step()
        return self.output_transform(x, y, y_pred, loss)
