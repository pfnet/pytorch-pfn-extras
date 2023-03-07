import contextlib
from typing import Any, Dict, Generator, Union

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
        autocast_options: Union[bool, Dict[str, Any]],
        has_grad_scaler: bool,
    ) -> None:
        from pytorch_pfn_extras._torch_version import requires
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

        if not _amp_enabled:
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
            with torch.cuda.amp.autocast(self._options["enabled"]):  # type: ignore[no-untyped-call]
                yield
        else:
            with torch.autocast(**self._options):  # type: ignore[no-untyped-call,attr-defined]
                yield
