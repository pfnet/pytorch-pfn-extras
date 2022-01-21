import multiprocessing as mp
import random
import time

import pytorch_pfn_extras as ppe
from torch.utils.tensorboard import SummaryWriter


def subprocess(num: int):
    ppe_extensions = [
        ppe.training.extensions.LogReport(),
        ppe.training.extensions.PlotReport(
            y_keys=["train/loss", "val/loss"],
            x_key="epoch",
            file_name="loss{}.png".format(num),
        ),
    ]
    writer = ppe.writing.SimpleWriter(out_dir="save/logs-{}".format(num))
    manager = ppe.training.ExtensionsManager(
        models={},
        optimizers={},
        max_epochs=3,
        extensions=ppe_extensions,
        writer=writer,
        iters_per_epoch=1,
    )
    tb_writer = SummaryWriter(log_dir="save/tensorboard-{}".format(num))

    @ppe.training.make_extension(trigger=(1, "epoch"))
    def tensorboard_writer(manager):
        m = manager.observation
        tb_writer.add_scalars(
            "loss",
            {"train/loss": m["train/loss"], "val/loss": m["val/loss"]},
            manager.epoch,
        )

    manager.extend(tensorboard_writer)

    for _ in range(3):
        with manager.run_iteration():
            ppe.reporting.report({"train/loss": random.random()})
            ppe.reporting.report({"val/loss": random.random()})
    time.sleep(1)
    return


def main():
    process = []
    for i in range(2):
        p = mp.Process(target=subprocess, args=(i,))
        process.append(p)
    for p in process:
        p.start()
    for p in process:
        p.join()
        p.close()


if __name__ == "__main__":
    main()
