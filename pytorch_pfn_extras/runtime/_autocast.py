import contextlib
from typing import Any, Dict, Generator
from pytorch_pfn_extras._torch_version import requires

_cuda_amp_available = False

try:
    import torch.cuda.amp
    _cuda_amp_available = torch.cuda.is_available() and hasattr(
        torch.cuda.amp, 'autocast')
except ImportError:
    pass


class _AutocastManager:
    def __init__(
        self,
        autocast_options: Dict[str, Any],
        has_grad_scaler: bool,
    ) -> None:
        options = {}
        if isinstance(autocast_options, dict):
            options.update(autocast_options)
        else:
            assert isinstance(autocast_options, bool)
            # Default to old behavior
            options = {
                "device_type": "cuda" if autocast_options else "cpu",
                "enabled": autocast_options
            }
        self._options = options
        self._use_old_ac = not requires("1.10.0")
        if (
            self._use_old_ac
            and self._options.get("enabled", False)
            and self._options.get("device_type", "cuda") != "cuda"
        ):
            raise RuntimeError("Autocast only work with CUDA devices for PyTorch 1.9")

        if not _cuda_amp_available:
            if (
                has_grad_scaler
                or self._options["device_type"] == "cuda"
            ):
                raise RuntimeError('Requested AMP features but torch.cuda.amp'
                                   ' is not enabled')

    @contextlib.contextmanager
    def autocast(self, enabled: bool = True) -> Generator[None, None, None]:
        # CUDA Availability was checked in Runtime Constructor
        if self._use_old_ac:
            with torch.cuda.amp.autocast(**self._options):  # type: ignore[no-untyped-call]
                yield
        else:
            with torch.autocast(**self._options):  # type: ignore[no-untyped-call,attr-defined]
                yield
