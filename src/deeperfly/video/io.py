"""Top-level frame I/O: ``read_video`` / ``read_images`` / ``write_mp4``.

Thin dispatchers over the backend registry (:mod:`deeperfly.video.base`). The
``backend`` argument selects an implementation (``"auto"`` picks the first
installed one); ``device`` lets GPU-capable readers keep frames on the device.
"""

from __future__ import annotations

import glob
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from jaxtyping import Float

from .base import is_gpu_device, select_reader, select_writer, to_numpy

log = logging.getLogger("deeperfly.video")

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
_JPEG_EXTS = (".jpg", ".jpeg")
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


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
    out = frames if is_gpu_device(device) else to_numpy(frames)
    log.info(
        "read video %s via '%s' backend -> %d frames %dx%d (device=%s)",
        Path(path).name,
        reader.name,
        out.shape[0],
        out.shape[1],
        out.shape[2],
        device,
    )
    return out


def list_image_files(pattern: str | Path) -> list[Path]:
    """Sorted image files for a directory or glob pattern (by name)."""
    p = Path(pattern)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    else:
        files = sorted(Path(f) for f in glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"no images matched {pattern!r}")
    return files


def _subset(files: list[Path], indices, start, stop, step) -> list[Path]:
    """Select frames by explicit ``indices`` or a ``range(start, stop, step)`` slice."""
    if indices is not None:
        return [files[int(i)] for i in indices]
    return files[start : stop if stop is not None else len(files) : step]


def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Coerce a decoded image to ``(H, W, 3)`` uint8 (grayscale broadcast, alpha dropped)."""
    if arr.ndim == 2:  # grayscale (H, W) -> (H, W, 1)
        arr = arr[..., None]
    if arr.shape[-1] == 1:  # single channel -> 3 (broadcast, not a width slice!)
        arr = np.repeat(arr, 3, axis=-1)
    arr = arr[..., :3]  # drop alpha / extra channels
    return arr if arr.dtype == np.uint8 else np.clip(arr, 0, 255).astype(np.uint8)


def _workers(n: int, workers: int | None) -> int:
    return max(1, min(n, workers or (os.cpu_count() or 4)))


def _read_images_cpu(files: list[Path], workers: int | None) -> np.ndarray:
    """Parallel CPU decode -> ``(T, H, W, 3)`` uint8 NumPy."""
    import imageio.v3 as iio

    def one(f: Path) -> np.ndarray:
        return _to_rgb_uint8(np.asarray(iio.imread(f)))

    with ThreadPoolExecutor(max_workers=_workers(len(files), workers)) as pool:
        frames = list(pool.map(one, files))
    return np.stack(frames)


def _read_images_gpu(files: list[Path], device: str, workers: int | None):
    """GPU decode -> ``(T, H, W, 3)`` uint8 ``torch.Tensor`` on ``device``.

    JPEGs go through nvJPEG (``torchvision.io.decode_jpeg`` on CUDA) when that
    build is available; PNG/other formats (or a torchvision without GPU JPEG) are
    decoded on the CPU in parallel and moved to the device. Either way the result
    is a GPU tensor, ready for :func:`~deeperfly.video.to_jax` (zero-copy).
    """
    import torch
    from torchvision.io import ImageReadMode, decode_jpeg, read_file

    n_workers = _workers(len(files), workers)
    if all(f.suffix.lower() in _JPEG_EXTS for f in files):
        try:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                raw = list(
                    pool.map(lambda f: read_file(str(f)), files)
                )  # parallel reads
            decoded = decode_jpeg(raw, mode=ImageReadMode.RGB, device=device)  # nvJPEG
            return torch.stack(decoded).permute(0, 2, 3, 1).contiguous()  # (T,H,W,3)
        except RuntimeError:
            pass  # no GPU JPEG support in this torchvision build -> CPU fallback
    return torch.as_tensor(_read_images_cpu(files, workers), device=device)


def read_images(
    pattern: str | Path,
    *,
    device: str = "cpu",
    indices: list[int] | None = None,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    workers: int | None = None,
) -> Float[np.ndarray, "T H W 3"]:
    """Read a directory (or glob) of images, sorted by name, into ``(T, H, W, 3)``.

    Decoding is parallel across threads (JPEG/PNG decoders release the GIL), so
    throughput scales with cores. Grayscale frames are broadcast to 3 channels and
    alpha is dropped; the result is ``uint8`` RGB.

    Parameters
    ----------
    device
        ``"cpu"`` returns NumPy; a CUDA device (``"cuda"`` / ``"cuda:0"``) returns
        a GPU-resident ``torch.Tensor`` (JPEGs via nvJPEG when available), ready
        for :func:`~deeperfly.video.to_jax`.
    indices, start, stop, step
        Select a subset, mirroring :func:`read_video` (``indices`` wins).
    workers
        Decode thread count (default: number of CPUs, capped at the frame count).
    """
    files = _subset(list_image_files(pattern), indices, start, stop, step)
    if not files:
        raise ValueError("no frames selected (check indices / start:stop:step)")
    out = (
        _read_images_gpu(files, device, workers)
        if is_gpu_device(device)
        else _read_images_cpu(files, workers)
    )
    log.info(
        "read %d images (imageio) -> %d frames %dx%d (device=%s)",
        len(files),
        out.shape[0],
        out.shape[1],
        out.shape[2],
        device,
    )
    return out


def read_frames(
    path: str | Path,
    *,
    backend: str = "auto",
    device: str = "cpu",
    indices: list[int] | None = None,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    workers: int | None = None,
) -> Float[np.ndarray, "T H W 3"]:
    """Read frames from a video file **or** an image sequence into ``(T, H, W, 3)``.

    Dispatches on ``path``: a video file (``.mp4`` / ``.avi`` / ``.mov`` ...) goes
    to :func:`read_video` (``backend`` selects the decoder); a directory or glob
    of images goes to :func:`read_images` (``workers`` sets decode parallelism).
    Both honor ``device`` (CPU NumPy or GPU tensor) and the same frame selection.
    """
    p = Path(path)
    if p.is_file() and p.suffix.lower() in _VIDEO_EXTS:
        return read_video(
            p,
            backend=backend,
            device=device,
            start=start,
            stop=stop,
            step=step,
            indices=indices,
        )
    return read_images(
        path,
        device=device,
        indices=indices,
        start=start,
        stop=stop,
        step=step,
        workers=workers,
    )


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
    log.info(
        "writing %s via '%s' backend: %d frames %dx%d @ %g fps",
        Path(path).name,
        writer.name,
        frames.shape[0],
        frames.shape[1],
        frames.shape[2],
        fps,
    )
    writer.write(frames, path, fps=fps, codec=codec, **kwargs)
