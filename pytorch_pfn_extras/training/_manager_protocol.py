from typing import Dict, Mapping, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pytorch_pfn_extras.training.extension import Extension


class ExtensionsManagerProtocol:

    @property
    def iteration(self) -> int:
        raise NotImplementedError

    @property
    def epoch(self) -> int:
        raise NotImplementedError

    @property
    def epoch_detail(self) -> float:
        raise NotImplementedError

    @property
    def _iters_per_epoch(self) -> int:
        raise NotImplementedError

    @property
    def models(self) -> Dict[str, torch.nn.Module]:
        raise NotImplementedError

    @property
    def raw_models(self) -> Mapping[str, torch.nn.Module]:
        raise NotImplementedError

    @property
    def optimizers(self) -> Mapping[str, torch.optim.Optimizer]:
        raise NotImplementedError

    @property
    def elapsed_time(self) -> float:
        raise NotImplementedError

    @property
    def is_before_training(self) -> bool:
        raise NotImplementedError

    @property
    def stop_trigger(self) -> bool:
        raise NotImplementedError

    @property
    def out(self) -> str:
        raise NotImplementedError

    def get_extension(self, name: str) -> 'Extension':
        raise NotImplementedError
