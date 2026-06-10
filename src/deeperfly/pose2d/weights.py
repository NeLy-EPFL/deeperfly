"""Weight I/O for the PyTorch detector: read the original DeepFly2D checkpoint.

The released checkpoint is a plain PyTorch ``state_dict`` (possibly wrapped by
Lightning/DataParallel), so it loads directly into the
:class:`~deeperfly.pose2d.model.HourglassNet` -- no conversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import model as _model
from .model import HourglassNet, device


def _place(model: HourglassNet, dev: str | None) -> HourglassNet:
    """Move ``model`` to ``dev`` in eval mode; on CUDA, switch it to channels_last.

    ``channels_last`` keeps the logical NCHW shape but stores the weights NHWC in
    memory, so cuDNN picks its faster (Tensor-Core) conv kernels -- a CUDA-only win,
    so CPU/MPS keep the default layout. Gated by
    :data:`~deeperfly.pose2d.model.USE_CHANNELS_LAST` (read live, not at import).
    """
    target = dev or device()
    model = model.eval().to(target)
    if _model.USE_CHANNELS_LAST and torch.device(target).type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def state_dict_from_torch_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
    """Load a DeepFly2D checkpoint into a native ``HourglassNet`` state-dict.

    Strips Lightning/DataParallel ``module.`` / ``model.`` prefixes and returns
    plain NumPy arrays.

    Parameters
    ----------
    path
        Path to a ``.pth`` checkpoint.

    Returns
    -------
    dict of str to np.ndarray
        The cleaned state-dict (NumPy arrays keyed by native parameter name).
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
    ``sh8`` = 8 stacks), so the architecture always matches before the strict load.

    Parameters
    ----------
    checkpoint
        Path to a ``.pth`` checkpoint, or ``None`` for a freshly initialized model.
    dev
        Target device string (defaults to the backend's auto-selected device).

    Returns
    -------
    HourglassNet
        The detector model in eval mode on ``dev``.
    """
    from .detector import infer_num_stacks

    if checkpoint is None:
        return _place(HourglassNet(), dev)
    sd = state_dict_from_torch_checkpoint(checkpoint)
    model = HourglassNet(num_stacks=infer_num_stacks(sd))
    model.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()}, strict=True)
    return _place(model, dev)
