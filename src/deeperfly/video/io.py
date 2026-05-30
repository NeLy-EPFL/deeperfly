"""Top-level frame I/O: ``read_video`` / ``read_images`` / ``write_mp4``.

Thin dispatchers over the backend registry (:mod:`deeperfly.video.base`). The
``backend`` argument selects an implementation (``"auto"`` picks the first
installed one); ``device`` lets GPU-capable readers keep frames on the device.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from jaxtyping import Float

from .base import is_gpu_device, select_reader, select_writer, to_numpy

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def read_video(
    path: str | Path,
    *,
    backend: str = "auto",
    device: str = "cpu",
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    indices: list[int] | None = None,
) -> Float[np.ndarray, "T H W 3"]:
    """Decode a video file to ``(T, H, W, 3)`` RGB frames.

    Parameters
    ----------
    backend
        ``"auto"`` | ``"imageio"`` | ``"opencv"`` | ``"pyav"`` | ``"decord"`` |
        ``"video_reader_rs"`` | ``"torchcodec"`` | ``"pynvvideocodec"`` |
        ``"dali"``. ``"auto"`` picks the first installed backend appropriate for
        ``device``.
    device
        ``"cpu"`` (NumPy result) or a CUDA device such as ``"cuda"`` /
        ``"cuda:0"`` (GPU-resident ``torch.Tensor`` result, GPU backends only).
    start, stop, step
        Sequential frame slice, like ``range(start, stop, step)``.
    indices
        Explicit frame indices for random access (overrides ``start/stop/step``).
        Seek-capable backends fetch just these frames; others decode up to
        ``max(indices)`` and gather.
    """
    reader = select_reader(backend, device=device)
    frames = reader.read(
        path, device=device, start=start, stop=stop, step=step, indices=indices
    )
    return frames if is_gpu_device(device) else to_numpy(frames)


def read_images(pattern: str | Path) -> Float[np.ndarray, "T H W 3"]:
    """Read a directory (or glob) of images, sorted by name, into ``(T, H, W, 3)``."""
    import imageio.v2 as imageio

    p = Path(pattern)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    else:
        files = sorted(Path(f) for f in glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"no images matched {pattern!r}")
    return np.stack([np.asarray(imageio.imread(f))[..., :3] for f in files])


def write_mp4(
    frames: Float[np.ndarray, "T H W 3"],
    path: str | Path,
    fps: float = 30.0,
    *,
    backend: str = "auto",
    codec: str | None = None,
    **kwargs,
) -> None:
    """Write ``(T, H, W, 3)`` frames to an MP4.

    ``frames`` may be NumPy or a GPU tensor; non-``uint8`` input is clipped to
    ``[0, 255]``. ``backend`` is ``"auto"`` | ``"imageio"`` | ``"pyav"`` |
    ``"opencv"``; ``codec`` overrides the backend default (libx264 for
    imageio/pyav, the ``mp4v`` fourcc for opencv).
    """
    writer = select_writer(backend)
    frames = to_numpy(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    writer.write(frames, path, fps=fps, codec=codec, **kwargs)
