"""The two interchangeable backends for the 2D detector.

deeperfly ships the stacked-hourglass detector in two frameworks behind one
interface:

* :mod:`~deeperfly.pose2d.backends.jax` -- the **default**. An Equinox port that
  runs the published ``sh8`` weights as a pure JAX PyTree (``jit`` / ``vmap``-able)
  and is the faster backend on GPU (see ``dev/bench_pose2d.py``).
* :mod:`~deeperfly.pose2d.backends.torch` -- a faithful copy of the original
  DeepFly2D network that loads the released DeepFly2D weights directly, with no
  conversion. A reference implementation and a path for users already on PyTorch.

Each backend exposes the same trio -- ``HourglassNet``, :func:`load_model` and
``predict_heatmaps`` -- and the same heatmap contract, so
:func:`deeperfly.pose2d.inference.detect` is identical whichever you pick. The
two backends are imported lazily, so selecting one never imports the other
framework (and importing :mod:`deeperfly.pose2d` never imports torch).
"""

from __future__ import annotations

import os
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
    each backend falls back to a freshly initialized model.
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


def detector_device(model) -> str:
    """Device the detector's parameters live on (e.g. ``"cuda:0"``, ``"cpu"``).

    Dispatches on the model's backend like :func:`predict_heatmaps` -- the JAX
    forward runs wherever its array leaves live (:func:`backends.jax.to_device`),
    so we read a leaf's device; the torch forward runs on its parameters' device.
    Lets callers log where 2D inference actually runs. ``"cpu"`` if no array leaf
    is found.
    """
    if type(model).__module__.startswith(f"{__name__}.torch"):
        return str(next(model.parameters()).device)
    import jax

    for leaf in jax.tree_util.tree_leaves(model):
        if hasattr(leaf, "devices"):  # a jax.Array leaf
            return str(next(iter(leaf.devices())))
    return "cpu"


# Measured marginal GPU memory of one sh8-hourglass forward at 256x512 (the
# per-image slope; jax ~80 MB, torch ~61 MB -- see dev/bench_pose2d.py). The large
# fixed cost is transient cuDNN workspace that shrinks when memory is tight, so it
# is not counted here; `safety` below leaves headroom for it plus the weights.
_FWD_BYTES_PER_IMAGE = 90 * 1024**2
_REF_PIXELS = 256 * 512


def gpu_memory_bytes(device=None) -> int | None:
    """Usable accelerator memory (bytes), or ``None`` when running on CPU.

    Reports the CUDA device's total memory, or -- on Apple Silicon -- Metal's
    (MPS) recommended working-set size, since there the GPU shares the system's
    unified memory. Uses torch (a core dependency) to query the physical device,
    so it reports the same number whichever detector backend is in use.
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


def _xla_mem_fraction() -> float:
    """Fraction of GPU memory the JAX pool may use, for the batch-fit budget.

    ``deeperfly run`` sets ``XLA_PYTHON_CLIENT_MEM_FRACTION`` (to share the card
    with on-device frames); JAX's own default preallocation is ~0.75. Sizing a
    batch against the *whole* card instead of this pool overshoots it and the
    forward OOMs.
    """
    try:
        return float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75"))
    except ValueError:
        return 0.75


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
    memory JAX can actually use -- ``total VRAM`` times :func:`_xla_mem_fraction`,
    **not** the whole card -- by it, clamped to ``[min_batch, max_batch]``. Without
    a GPU it returns ``min_batch``.

    Budgeting against the pool matters: ``deeperfly run`` caps JAX at half the card
    (``XLA_PYTHON_CLIENT_MEM_FRACTION``), so a batch sized to the *full* card
    overshoots that pool and the forward OOMs -- XLA then retries on a slow
    fragmented path (the ~2x slowdown that motivated this). The cap also matters:
    for this 8-stack network throughput saturates at a small batch on a fast GPU
    (bigger batches don't help -- see ``dev/bench_pose2d.py``), so sizing is to
    *fit* memory and avoid OOM, not to chase speed.
    """
    total = gpu_memory_bytes(device)
    if total is None:
        return min_batch
    h, w = image_hw
    per_image = _FWD_BYTES_PER_IMAGE * (h * w) / _REF_PIXELS
    fit = int(total * _xla_mem_fraction() * safety / per_image)
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
    "detector_device",
    "infer_num_stacks",
    "gpu_memory_bytes",
    "auto_batch_size",
]
