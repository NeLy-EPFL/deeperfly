"""The torch-free seam in front of the 2D detectors.

What is left here is what every detector class shares and no class owns: where its
parameters live (:func:`detector_device`) and how much GPU memory there is to size a
batch against (:func:`gpu_memory_bytes`). Both answer their question without importing
torch at module scope, which is the whole point of the seam -- :mod:`deeperfly.pose2d`
imports this module eagerly, and :mod:`deeperfly.pose2d.runtime` (which does import
torch) must stay behind a lazy import.

Setting the forward precision is deliberately NOT here: it writes to a live
``nn.Module``, so every caller already holds torch by the time it asks, and a seam in
front of it would front nothing. Use :func:`deeperfly.pose2d.runtime.set_precision`.

The forward and the decode are NOT here either. Both shipped classes own theirs -- the
dense HRNet because its heatmap field is padded past its input, the multiview transformer
because its soft-argmax lives inside the module -- so there is no shared implementation to
front, only a shared vocabulary.
"""

from __future__ import annotations

import logging

log = logging.getLogger("deeperfly")


def detector_device(model) -> str:
    """Device the detector's parameters live on (e.g. ``"cuda:0"``, ``"cpu"``).

    Lets callers log where 2D inference runs and tells the orchestration where to
    upload frames.

    Parameters
    ----------
    model
        The detector.

    Returns
    -------
    str
        The device string (``"cpu"`` for a parameterless model).
    """
    params = getattr(model, "parameters", None)
    if params is None:
        return "cpu"
    try:
        return str(next(params()).device)
    except StopIteration:
        return "cpu"


def gpu_memory_bytes(device=None) -> int | None:
    """Usable accelerator memory (bytes), or ``None`` when running on CPU.

    The CUDA device's total memory, or -- on Apple Silicon -- Metal's (MPS)
    recommended working-set size (the GPU shares unified memory there).

    Parameters
    ----------
    device
        An optional CUDA device string to query (defaults to device 0).

    Returns
    -------
    int or None
        The memory in bytes, or ``None`` on CPU / when unavailable.
    """
    try:
        import torch

        if torch.cuda.is_available():
            idx = 0
            if device is not None and str(device).startswith("cuda"):
                parts = str(device).split(":")
                idx = int(parts[1]) if len(parts) > 1 and parts[1] else 0
            return int(torch.cuda.get_device_properties(idx).total_memory)
        if torch.backends.mps.is_available():
            return int(torch.mps.recommended_max_memory())
        return None
    except Exception as exc:  # noqa: BLE001
        log.debug("GPU memory probe failed: %s", exc)
        return None
