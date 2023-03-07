import contextlib
from typing import Any, Dict, Generator

_amp_enabled = False

try:
    import torch.cuda.amp
    _amp_enabled = torch.cuda.is_available() and hasattr(
        torch.cuda.amp, 'autocast')
except ImportError:
    pass


class _AutocastManager:
    def __init__(
        self,
        autocast_options: Dict[str, Any],
    ) -> None:
        from pytorch_pfn_extras._torch_version import requires
        self._options = autocast_options
        self._use_old_ac = not requires("1.10.0")
        if (
            self._use_old_ac
            and self._options.get("enabled", False)
            and self._options.get("device_type", "cuda") != "cuda"
        ):
            raise RuntimeError("Autocast only work with CUDA devices for PyTorch 1.9")

    @contextlib.contextmanager
    def autocast(self, enabled: bool = True) -> Generator[None, None, None]:
        # CUDA Availability was checked in Runtime Constructor
        if self._use_old_ac:
            with torch.cuda.amp.autocast(self._options["enabled"]):  # type: ignore[no-untyped-call]
                yield
        else:
            with torch.autocast(**self._options):  # type: ignore[no-untyped-call,attr-defined]
                yield
