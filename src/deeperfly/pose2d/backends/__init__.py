"""The PyTorch 2D detector backend and its shared helpers.

deeperfly ships the stacked-hourglass detector in PyTorch
(:mod:`~deeperfly.pose2d.backends.torch`): a faithful copy of the original
DeepFly2D network that loads the released DeepFly2D weights directly, with no
conversion. It runs on CUDA (NVIDIA) and Metal/MPS (Apple Silicon) automatically.

This package exposes the detector behind a small interface -- :func:`load_detector`,
:func:`predict_heatmaps`, :func:`detector_device` -- so the shared orchestration in
:mod:`deeperfly.pose2d.inference` never touches the backend directly, plus the
GPU-sizing helpers (:func:`gpu_memory_bytes`, :func:`auto_batch_size`) and
:func:`infer_num_stacks`. The backend is imported lazily, so importing
:mod:`deeperfly.pose2d` never imports torch.
"""

from __future__ import annotations

import numpy as np


def load_detector(checkpoint=None, **kwargs):
    """Load the PyTorch detector, optionally from ``checkpoint`` (a ``.pth``).

    Extra keyword arguments are forwarded to the backend's ``load_model`` (e.g.
    ``dev``). With ``checkpoint=None`` a freshly initialized model is returned.
    """
    from . import torch as backend

    return backend.load_model(checkpoint, **kwargs)


def predict_heatmaps(model, inputs: np.ndarray) -> np.ndarray:
    """Final-stack heatmaps for ``(N, 3, H, W)`` inputs, always as NumPy.

    Returns NumPy so downstream :func:`~deeperfly.pose2d.inference.heatmap_to_points`
    decoding is independent of where the forward ran.
    """
    from . import torch as backend

    return np.asarray(backend.predict_heatmaps(model, inputs))


def set_precision(model, precision: str = "float32") -> None:
    """Set the detector forward precision: ``"float32"`` (default) or ``"float16"``.

    ``"float16"`` runs the forward under CUDA autocast (faster, negligible keypoint
    drift); it is a no-op on CPU/MPS. Stored on the model, so the next
    :func:`predict_heatmaps` honors it.
    """
    from . import torch as backend

    backend.set_precision(model, precision)


def detector_device(model) -> str:
    """Device the detector's parameters live on (e.g. ``"cuda:0"``, ``"cpu"``).

    Lets callers log where 2D inference actually runs, and tells the orchestration
    where to upload frames. Falls back to ``"cpu"`` for a parameterless model (a
    stub, or one with no parameters).
    """
    params = getattr(model, "parameters", None)
    if params is None:
        return "cpu"
    try:
        return str(next(params()).device)
    except StopIteration:
        return "cpu"


# Measured marginal GPU memory of one sh8-hourglass forward at 256x512 (the
# per-image slope; ~61 MB for torch -- see dev/bench_video.py). The large fixed
# cost is transient cuDNN workspace that shrinks when memory is tight, so it is
# not counted here; `safety` below leaves headroom for it plus the weights.
_FWD_BYTES_PER_IMAGE = 90 * 1024**2
_REF_PIXELS = 256 * 512


def gpu_memory_bytes(device=None) -> int | None:
    """Usable accelerator memory (bytes), or ``None`` when running on CPU.

    Reports the CUDA device's total memory, or -- on Apple Silicon -- Metal's
    (MPS) recommended working-set size, since there the GPU shares the system's
    unified memory. Uses torch to query the physical device.
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
    except Exception:
        return None


def auto_batch_size(
    image_hw: tuple[int, int] = (256, 512),
    *,
    device=None,
    safety: float = 0.5,
    min_batch: int = 1,
    max_batch: int = 32,
) -> int:
    """Pick a detector batch size (images per forward) that fits the GPU's VRAM.

    Scales the measured per-image forward cost (:data:`_FWD_BYTES_PER_IMAGE`, at
    256x512) by the actual ``image_hw`` and divides a ``safety`` fraction of the
    card's total memory by it, clamped to ``[min_batch, max_batch]``. Without a
    GPU it returns ``min_batch``.

    The ``safety`` margin leaves headroom for the weights and the transient cuDNN
    workspace, so the forward fits rather than OOMing. The cap also matters: for
    this 8-stack network throughput saturates at a small batch on a fast GPU
    (bigger batches don't help -- see ``dev/bench_video.py``), so sizing is to
    *fit* memory, not to chase speed.
    """
    total = gpu_memory_bytes(device)
    if total is None:
        return min_batch
    h, w = image_hw
    per_image = _FWD_BYTES_PER_IMAGE * (h * w) / _REF_PIXELS
    fit = int(total * safety / per_image)
    return max(min_batch, min(max_batch, fit))


def infer_num_stacks(state_dict) -> int:
    """Number of hourglass stacks in a ``state_dict`` (counts ``score.{i}.weight``).

    Used by the backend's loader so the architecture matches the checkpoint before
    a strict load. The published checkpoint is ``sh8`` (8 stacks), but the count is
    derived from the weights so any depth round-trips.
    """
    n = 0
    while f"score.{n}.weight" in state_dict:
        n += 1
    if n == 0:
        raise KeyError(
            "no 'score.{i}.weight' keys found; not a HourglassNet state_dict"
        )
    return n


__all__ = [
    "load_detector",
    "predict_heatmaps",
    "detector_device",
    "set_precision",
    "infer_num_stacks",
    "gpu_memory_bytes",
    "auto_batch_size",
]
