import contextlib
from typing import Any, Dict, Generator

_cuda_amp_available = False

try:
    import torch.cuda.amp

    _cuda_amp_available = torch.cuda.is_available() and hasattr(
        torch.cuda.amp, "autocast"
    )
except ImportError:
    pass


class _AutocastManager:
    def __init__(
        self,
        autocast_options: Dict[str, Any],
        has_grad_scaler: bool,
    ) -> None:
        autocast_options = autocast_options.copy()
        self._enabled = autocast_options.pop("enabled", True)
        self._device_type = autocast_options.pop("device_type", "cuda")
        self._options = autocast_options

        if not _cuda_amp_available:
            if has_grad_scaler or (
                self._enabled and self._device_type == "cuda"
            ):
                raise RuntimeError(
                    "Requested AMP features but torch.cuda.amp"
                    " is not enabled"
                )

    @contextlib.contextmanager
    def autocast(self, enabled: bool = True) -> Generator[None, None, None]:
        # CUDA Availability was checked in Runtime Constructor
        with torch.autocast(self._device_type, enabled=self._enabled, **self._options):  # type: ignore[no-untyped-call,attr-defined]
            yield
