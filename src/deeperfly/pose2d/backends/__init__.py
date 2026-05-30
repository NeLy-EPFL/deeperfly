"""The two interchangeable backends for the 2D detector.

deeperfly ships the stacked-hourglass detector in two frameworks behind one
interface:

* :mod:`~deeperfly.pose2d.backends.jax` -- the **default**. An Equinox port that
  runs the published ``sh8`` weights as a pure JAX PyTree (``jit`` / ``vmap``-able)
  and is the faster backend on GPU (see ``dev/bench_pose2d.py``).
* :mod:`~deeperfly.pose2d.backends.torch` -- a faithful copy of the original
  DeepFly2D network that loads the released ``.tar`` weights directly, with no
  conversion. A reference implementation and a path for users already on PyTorch.

Each backend exposes the same trio -- ``HourglassNet``, :func:`load_model` and
``predict_heatmaps`` -- and the same heatmap contract, so
:func:`deeperfly.pose2d.inference.detect` is identical whichever you pick. The
two backends are imported lazily, so selecting one never imports the other
framework (and importing :mod:`deeperfly.pose2d` never imports torch).
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

#: Available detector backends, preferred first (``jax`` is faster on GPU).
BACKENDS = ("jax", "torch")
DEFAULT_BACKEND = "jax"


def _backend(name: str) -> ModuleType:
    """Import a backend subpackage by name (lazy, so the other never loads)."""
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {BACKENDS}")
    if name == "torch":
        from . import torch as module
    else:
        from . import jax as module
    return module


def load_detector(backend: str = DEFAULT_BACKEND, checkpoint=None, **kwargs):
    """Load a detector for ``backend`` (``"jax"`` or ``"torch"``).

    Extra keyword arguments are forwarded to the backend's ``load_model`` (e.g.
    ``key`` / ``num_stacks`` for jax, ``dev`` for torch). With ``checkpoint=None``
    each backend falls back to a freshly initialised model.
    """
    return _backend(backend).load_model(checkpoint, **kwargs)


def predict_heatmaps(model, inputs: np.ndarray) -> np.ndarray:
    """Final-stack heatmaps for ``(N, 3, H, W)`` inputs, dispatching on ``model``.

    Picks the backend that owns ``model`` (by its module) and always returns
    NumPy, so downstream :func:`~deeperfly.pose2d.inference.heatmap_to_points`
    decoding is backend-agnostic.
    """
    name = "torch" if type(model).__module__.startswith(f"{__name__}.torch") else "jax"
    return np.asarray(_backend(name).predict_heatmaps(model, inputs))


# Measured marginal GPU memory of one sh8-hourglass forward at 256x512 (the
# per-image slope; jax ~80 MB, torch ~61 MB -- see dev/bench_pose2d.py). The large
# fixed cost is transient cuDNN workspace that shrinks when memory is tight, so it
# is not counted here; `safety` below leaves headroom for it plus the weights.
_FWD_BYTES_PER_IMAGE = 90 * 1024**2
_REF_PIXELS = 256 * 512


def gpu_memory_bytes(device=None) -> int | None:
    """Total memory (bytes) of the CUDA device, or ``None`` when there is no GPU.

    Uses torch (a core dependency) to query the physical device, so it reports the
    same number whichever detector backend is in use.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        idx = 0
        if device is not None:
            parts = str(device).split(":")
            idx = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        return int(torch.cuda.get_device_properties(idx).total_memory)
    except Exception:
        return None


def auto_batch_size(
    image_hw: tuple[int, int] = (256, 512),
    *,
    device=None,
    safety: float = 0.5,
    min_batch: int = 1,
    max_batch: int = 64,
) -> int:
    """Pick a detector batch size (images per forward) that fits the GPU's VRAM.

    Scales the measured per-image forward cost (:data:`_FWD_BYTES_PER_IMAGE`, at
    256x512) by the actual ``image_hw`` and divides a ``safety`` fraction of total
    VRAM by it, clamped to ``[min_batch, max_batch]``. Without a GPU it returns
    ``min_batch``.

    The cap matters: for this 8-stack network throughput saturates at a small
    batch on a fast GPU (bigger batches don't help -- see ``dev/bench_pose2d.py``),
    so the point of sizing is to *fit* memory on smaller GPUs and avoid OOM, not
    to chase speed by going ever larger.
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

    Shared by both backends' loaders so the architecture matches the checkpoint
    before a strict load. The published checkpoint is ``sh8`` (8 stacks), but the
    count is derived from the weights so any depth round-trips.
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
    "BACKENDS",
    "DEFAULT_BACKEND",
    "load_detector",
    "predict_heatmaps",
    "infer_num_stacks",
    "gpu_memory_bytes",
    "auto_batch_size",
]
