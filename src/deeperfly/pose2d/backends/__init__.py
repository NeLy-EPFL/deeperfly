"""The PyTorch 2D detector backend and its shared helpers.

deeperfly ships the stacked-hourglass detector in PyTorch
(:mod:`~deeperfly.pose2d.backends.torch`): a faithful copy of the DeepFly2D
network that loads the released weights directly, running on CUDA (NVIDIA) and
Metal/MPS (Apple Silicon) automatically.

The detector sits behind a small interface (:func:`load_detector`,
:func:`predict_heatmaps`, :func:`detector_device`), so the orchestration in
:mod:`deeperfly.pose2d.inference` never touches the backend directly, plus the
GPU-memory helper (:func:`gpu_memory_bytes`) and
:func:`infer_num_stacks`. The backend imports lazily, so importing
:mod:`deeperfly.pose2d` never imports torch.
"""

from __future__ import annotations

import numpy as np


def load_detector(checkpoint=None, **kwargs):
    """Load the PyTorch detector, optionally from ``checkpoint`` (a ``.pth``).

    Parameters
    ----------
    checkpoint
        Path to a ``.pth`` checkpoint, or ``None`` for a freshly initialized model.
    **kwargs
        Forwarded to the backend's ``load_model`` (e.g. ``dev``).

    Returns
    -------
    The loaded detector model.
    """
    from . import torch as backend

    return backend.load_model(checkpoint, **kwargs)


def predict_heatmaps(model, inputs: np.ndarray) -> np.ndarray:
    """Final-stack heatmaps for ``(N, 3, H, W)`` inputs, always as NumPy.

    Returns NumPy so downstream :func:`~deeperfly.pose2d.inference.heatmap_to_points`
    decoding is independent of where the forward ran. Used by the candidate path,
    which needs whole heatmaps; the plain detect path uses :func:`predict_points`.

    Parameters
    ----------
    model
        The detector.
    inputs
        Network inputs of shape ``(N, 3, H, W)``.

    Returns
    -------
    np.ndarray
        The final-stack heatmaps as host NumPy.
    """
    from . import torch as backend

    return np.asarray(backend.predict_heatmaps(model, inputs))


def predict_points(
    model, inputs: np.ndarray, *, method: str = "weighted", radius: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Fused forward + heatmap decode: normalized ``(N, J, 2)`` peaks and ``(N, J)`` conf.

    The arg-max decode runs on the forward's device, so only the small peak arrays
    cross to the host -- not the full heatmap, and not a host-side float64 arg-max.
    Equivalent to ``heatmap_to_points(predict_heatmaps(...))`` to float32 epsilon.

    Parameters
    ----------
    model
        The detector.
    inputs
        Network inputs of shape ``(N, 3, H, W)``.
    method, radius
        Sub-pixel refinement options (see
        :func:`~deeperfly.pose2d.inference.refine_peaks`).

    Returns
    -------
    xy : np.ndarray
        Normalized ``(N, J, 2)`` peaks.
    conf : np.ndarray
        Per-joint confidence of shape ``(N, J)``.
    """
    from . import torch as backend

    xy, conf = backend.predict_points(model, inputs, method=method, radius=radius)
    return np.asarray(xy), np.asarray(conf)


def set_precision(model, precision: str = "float32") -> None:
    """Set the detector forward precision: ``"float32"``, ``"float16"``, or ``"bfloat16"``.

    ``"float16"`` / ``"bfloat16"`` run under CUDA autocast (faster, negligible
    keypoint drift; bfloat16 trades a little speed for a wider, overflow-proof
    range); a no-op on CPU/MPS. Stored on the model, so the next forward honors it.

    Parameters
    ----------
    model
        The detector (the precision is stored on it).
    precision
        ``"float32"``, ``"float16"`` or ``"bfloat16"``.
    """
    from . import torch as backend

    backend.set_precision(model, precision)


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
    except Exception:
        return None


def infer_num_stacks(state_dict) -> int:
    """Number of hourglass stacks in a ``state_dict`` (counts ``score.{i}.weight``).

    Lets the loader match the architecture to the checkpoint before a strict load.
    The published checkpoint is ``sh8`` (8 stacks), but the count is derived from
    the weights so any depth round-trips.

    Parameters
    ----------
    state_dict
        A HourglassNet ``state_dict``.

    Returns
    -------
    int
        The number of hourglass stacks.

    Raises
    ------
    KeyError
        If no ``score.{i}.weight`` keys are present (not a HourglassNet state).
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
    "predict_points",
    "detector_device",
    "set_precision",
    "infer_num_stacks",
    "gpu_memory_bytes",
]
