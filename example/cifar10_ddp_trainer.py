import argparse
import multiprocessing
import tempfile
from typing import Any, Dict

import pytorch_pfn_extras as ppe
import pytorch_pfn_extras.training.extensions as ext
import torch
import torch.nn as nn
from pytorch_pfn_extras.engine import create_evaluator, create_trainer
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import ResNet, resnet50


def run_forkserver():
    """Change the subprocess launch mode from "fork" to "forkserver"

    In "fork" mode, contexts in the current process such as MPI/CUDA handlers
    are copied to worker processes launched by DataLoader(0<num_workers),
    which can cause of double-free like problems. Using a forkserver launched
    from a fresh process prevents the same resources to be shared to children.
    """
    multiprocessing.set_start_method("forkserver", force=True)
    p = multiprocessing.Process()
    p.start()
    p.join()


class TrainerModel(nn.Module):
    def __init__(self, model: ResNet, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.model.forward(x)
        loss = self.criterion.forward(out, y)
        ppe.reporting.report({"train/loss": loss.item()})
        return {"loss": loss}


class EvaluatorModel(nn.Module):
    def __init__(self, model: ResNet, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> None:
        out = self.model.forward(x)
        loss = self.criterion.forward(out, y)
        acc = (out.argmax(axis=1) == y).sum() / len(y)
        ppe.reporting.report(
            {
                "val/loss": loss.item(),
                "val/accuracy": acc.item(),
            }
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train a ResNet50 model on CIFAR-10 dataset "
        "using PyTorch and pytorch-pfn-extras.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument(
        "--epoch", "-e", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="output",
        help="Output directory to save the results",
    )
    parser.add_argument(
        "--n-retains", "-n", type=int, default=5, help="Number of snapshots to retain"
    )
    parser.add_argument(
        "--num-worker",
        type=int,
        default=0,
        help="Number of worker threads for data loading",
    )
    parser.add_argument(
        "--cifar-dir",
        type=str,
        default=None,
        help="Directory for CIFAR-10 dataset; downloads dataset if not provided",
    )
    parser.add_argument(
        "--no-autoload",
        action="store_true",
        help="Specify this option if you don't need automatic "
        "restart of training from the previous snapshot "
        "in the output directory.",
    )
    parser.add_argument(
        "--use-mnbn",
        action="store_true",
        help="Specify when using Multi-node BatchNorm.",
    )

    parser.add_argument(
        "--mixed-fp16",
        action="store_true",
        help="Use mixed precision (FP16) for training to improve computation speed and reduce memory usage.",
    )
    args = parser.parse_args()

    world_size, world_rank, local_rank = ppe.distributed.initialize_ompi_environment(
        backend="nccl", init_method="tcp"
    )
    torch.cuda.set_device(torch.device("cuda:{}".format(local_rank)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.cifar_dir is None:
        tmp_dir = tempfile.TemporaryDirectory()
        cifar_dir = tmp_dir.name
    else:
        cifar_dir = args.cifar_dir

    train = CIFAR10(
        cifar_dir,
        download=True,
        train=True,
        transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor()]
        ),
    )
    val = CIFAR10(
        cifar_dir, download=True, train=False, transform=transforms.ToTensor()
    )

    train_sampler: DistributedSampler[int] = DistributedSampler(train)
    val_sampler: DistributedSampler[int] = DistributedSampler(val, shuffle=False)

    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        num_workers=args.num_worker,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val,
        batch_size=args.test_batch_size,
        num_workers=args.num_worker,
        sampler=val_sampler,
    )

    model = resnet50(num_classes=10)
    if args.use_mnbn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    trainer_model = TrainerModel(model=model)
    evaluator_model = EvaluatorModel(model=model)
    distributed_trainer_model = ppe.nn.parallel.DistributedDataParallel(ppe.to(trainer_model, device))

    optimizer = torch.optim.Adam(distributed_trainer_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=args.epoch
    )

    default_trigger = (1, "epoch")
    extensions = [
        ppe.training.ExtensionEntry(
            ext.snapshot(
                n_retains=args.n_retains,
                autoload=not args.no_autoload,
                saver_rank=0,
            ),
            trigger=default_trigger,
        ),
        ppe.training.ExtensionEntry(
            ext.LRScheduler(scheduler),
            trigger=(1, "epoch"),
        ),
    ]
    if world_rank == 0:
        extensions += [
            ext.LogReport(trigger=default_trigger),  # type: ignore
            ext.ProgressBar(),  # type: ignore
            ext.PrintReport(  # type: ignore
                ["epoch", "iteration", "train/loss", "val/loss", "val/accuracy", "lr"]
            ),
            ppe.training.ExtensionEntry(
                ext.observe_lr(optimizer),
                trigger=default_trigger,
            ),
        ]
    trainer = create_trainer(
        models=distributed_trainer_model,
        optimizers=optimizer,
        max_epochs=args.epoch,
        extensions=extensions,
        out_dir=args.out,
        evaluator=(
            create_evaluator(
                ppe.to(evaluator_model, device),
                progress_bar=world_rank == 0,
                device=device,
            ),
            default_trigger,
        ),
        device=device,
        options={
            "autocast": True,
            "grad_scaler": GradScaler(),
        }
        if args.mixed_fp16
        else {},
    )

    trainer.run(train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    main()
