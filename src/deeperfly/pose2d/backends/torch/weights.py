"""Weight I/O for the PyTorch backend: read the original DeepFly2D checkpoint.

The released checkpoint is a plain PyTorch ``state_dict`` (possibly wrapped by
Lightning/DataParallel), so this backend loads it directly into the
:class:`~deeperfly.pose2d.backends.torch.model.HourglassNet` -- no conversion.
:func:`state_dict_from_torch_checkpoint` is also what the JAX backend's
``convert_state_dict`` consumes when producing a native ``.eqx`` checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import HourglassNet, device


def state_dict_from_torch_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
    """Load a DeepFly2D checkpoint into a native ``HourglassNet`` state-dict.

    Strips Lightning/DataParallel ``module.`` / ``model.`` prefixes and returns
    plain NumPy arrays (so it doubles as the source for the JAX conversion).
    """
    raw = torch.load(path, map_location="cpu", weights_only=True)
    sd = raw["state_dict"] if "state_dict" in raw else raw
    out: dict[str, np.ndarray] = {}
    for k, v in sd.items():
        k = k[len("module.") :] if k.startswith("module.") else k
        k = k[len("model.") :] if k.startswith("model.") else k
        out[k] = v.detach().cpu().numpy()
    return out


def load_model(
    checkpoint: str | Path | None = None, *, dev: str | None = None
) -> HourglassNet:
    """Build the DeepFly2D net and (optionally) load the original DeepFly2D weights.

    The number of stacks is taken from the checkpoint (the published weights are
    ``sh8`` = 8 stacks), so the architecture always matches before the strict
    load. With ``checkpoint=None`` a freshly initialized model is returned.
    """
    from .. import infer_num_stacks  # shared across backends, torch-free import

    if checkpoint is None:
        return HourglassNet().eval().to(dev or device())
    sd = state_dict_from_torch_checkpoint(checkpoint)
    model = HourglassNet(num_stacks=infer_num_stacks(sd)).eval()
    model.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()}, strict=True)
    return model.to(dev or device())
