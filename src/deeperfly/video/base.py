"""Frame-array helpers and still-image reader resolution.

:func:`to_numpy` / :func:`to_torch` adapt decoded frames for callers that want a
NumPy array or a torch tensor. Image *sequences* (not video files) are decoded by
a small reader registry -- ``opencv`` (core) with an optional ``imageio`` fallback
-- resolved here via :func:`select_image_reader`. Video files are read and written
by PyAV directly in :mod:`deeperfly.video.io`.
"""

from __future__ import annotations

import importlib.util

import numpy as np


def _have(*modules: str) -> bool:
    """True if every module can be located without importing the heavy parts."""
    for mod in modules:
        try:
            if importlib.util.find_spec(mod) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def to_numpy(frames) -> np.ndarray:
    """Collapse decoded frames (NumPy / torch tensor) to a NumPy array.

    Parameters
    ----------
    frames
        A NumPy array or a torch tensor (or any array-like).

    Returns
    -------
    np.ndarray
        The frames as a host NumPy array.
    """
    if isinstance(frames, np.ndarray):
        return frames
    if hasattr(frames, "detach"):  # torch.Tensor
        return frames.detach().cpu().numpy()
    return np.asarray(frames)


def to_torch(frames):
    """Hand frames to torch, zero-copy where possible.

    A ``torch.Tensor`` passes through untouched, any other DLPack-capable array is
    wrapped via the DLPack protocol, and NumPy input (what the PyAV reader returns)
    is wrapped on the host via zero-copy ``torch.from_numpy``.

    Parameters
    ----------
    frames
        A torch tensor, a DLPack-capable array, or a NumPy array.

    Returns
    -------
    torch.Tensor
        The frames as a torch tensor (zero-copy where possible).
    """
    import torch

    if isinstance(frames, torch.Tensor):
        return frames
    if hasattr(frames, "__dlpack__"):  # DLPack-capable array
        return torch.from_dlpack(frames)
    return torch.from_numpy(to_numpy(frames))


# Still-image decode (image sequences, not video files). These aren't full backend
# classes -- the work is a per-file decode -- so the registry is just a preference
# order plus a name resolver. ``opencv`` is the core default; ``imageio`` is an
# optional broad-format fallback (see ``deeperfly.video.io._read_images_cpu``).
IMAGE_READ_ORDER = ("opencv", "imageio")
_IMAGE_READER_REQUIRES = {"opencv": ("cv2",), "imageio": ("imageio",)}


def list_image_readers() -> list[str]:
    """All known image-reader names (installed or not)."""
    return sorted(IMAGE_READ_ORDER)


def available_image_readers() -> list[str]:
    """Image-reader names whose dependencies are importable in this environment."""
    return sorted(n for n in IMAGE_READ_ORDER if _have(*_IMAGE_READER_REQUIRES[n]))


def select_image_reader(backend: str = "auto") -> str:
    """Resolve an image-reader name (or ``"auto"``).

    ``"auto"`` walks :data:`IMAGE_READ_ORDER` and returns the first installed reader
    (``opencv`` is a core dependency, so this is normally ``"opencv"``).

    Parameters
    ----------
    backend
        An image-reader name, or ``"auto"``.

    Returns
    -------
    str
        The resolved image-reader name.

    Raises
    ------
    ValueError
        If ``backend`` names no known image reader.
    RuntimeError
        If the named (or every auto-order) reader is unavailable.
    """
    if backend == "auto":
        for name in IMAGE_READ_ORDER:
            if _have(*_IMAGE_READER_REQUIRES[name]):
                return name
        raise RuntimeError(
            f"no image reader available; install one of {list(IMAGE_READ_ORDER)}"
        )
    if backend not in _IMAGE_READER_REQUIRES:
        raise ValueError(
            f"unknown image reader {backend!r}; choose from {list_image_readers()}"
        )
    if not _have(*_IMAGE_READER_REQUIRES[backend]):
        raise RuntimeError(
            f"image reader {backend!r} needs {_IMAGE_READER_REQUIRES[backend]}; "
            "install it (e.g. the optional 'imageio' extra)"
        )
    return backend
