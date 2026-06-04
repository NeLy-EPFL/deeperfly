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

from .base import (
    GPU_READ_ORDER,
    canonical_device,
    gpu_reader_candidates,
    is_gpu_device,
    mark_gpu_auto_failed,
    remember_gpu_reader,
    resolve_device,
    select_reader,
    select_writer,
    to_numpy,
)

log = logging.getLogger("deeperfly.video")

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
_JPEG_EXTS = (".jpg", ".jpeg")
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def _is_video_file(p: Path) -> bool:
    """Whether ``p`` is an existing video file (so :func:`read_frames` decodes it as
    a video rather than treating it as an image directory/glob/sequence)."""
    return p.is_file() and p.suffix.lower() in _VIDEO_EXTS


def read_video(
    path: str | Path,
    *,
    backend: str = "auto",
    device: str = "auto",
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    indices: list[int] | None = None,
) -> Float[np.ndarray, "T H W 3"]:
    """Decode a video file to ``(T, H, W, 3)`` RGB frames.

    Parameters
    ----------
    backend
        ``"auto"`` | ``"pyav"`` | ``"opencv"`` | ``"imageio"`` | ``"decord"`` |
        ``"video_reader_rs"`` | ``"torchcodec"`` | ``"dali"``. ``"auto"`` picks
        the fastest installed backend for the resolved ``device`` (GPU order
        leads with NVDEC; CPU order leads with in-process decoders -- pyav, the
        core default -- and keeps imageio last).
    device
        ``"auto"`` (the default) decodes on the GPU when one is present *and* a
        GPU backend works, else on the CPU -- and always returns host ``NumPy``
        (the portable path). ``"cpu"`` forces a NumPy result. A CUDA device
        (``"cuda"`` / ``"cuda:0"``) returns a GPU-resident ``torch.Tensor`` (ready
        for :func:`~deeperfly.video.to_jax`, zero-copy): with ``backend="auto"``
        it tries each installed GPU backend until one decodes and gracefully falls
        back to a CPU NumPy read if none can; with a forced ``backend`` it is
        strict and raises if that backend cannot decode on the GPU.
    start, stop, step
        Sequential frame slice, like ``range(start, stop, step)``.
    indices
        Explicit frame indices for random access (overrides ``start/stop/step``).
        Seek-capable backends fetch just these frames; others decode up to
        ``max(indices)`` and gather.
    """
    # "auto" device is the portable convenience path: it may hardware-decode on
    # the GPU but always hands back host NumPy. A CUDA device keeps the frames an
    # on-device tensor for zero-copy handoff. "auto" *backend* is best-effort -- a
    # GPU decode failure falls back to the CPU; a forced backend is strict.
    want_host = device == "auto"
    best_effort = backend == "auto"
    device = resolve_device(device, backend)
    read_kw = dict(start=start, stop=stop, step=step, indices=indices)

    if is_gpu_device(device):
        try:
            reader, frames = _read_on_gpu(path, backend, device, read_kw)
        except Exception as exc:
            if not best_effort:
                raise  # a forced GPU backend surfaces its failure
            log.warning(
                "GPU video decode unavailable (%s); using CPU decode for this and "
                "subsequent reads",
                exc,
            )
            mark_gpu_auto_failed()  # stop steering "auto" at the GPU this process
            device = "cpu"
            reader = select_reader(backend, device=device)
            frames = reader.read(path, device=device, **read_kw)
    else:
        reader = select_reader(backend, device=device)
        frames = reader.read(path, device=device, **read_kw)

    out = frames if is_gpu_device(device) and not want_host else to_numpy(frames)
    log.debug(  # per-read detail (one line per camera per window) -- only at -vv
        "read video %s via '%s' backend -> %d frames %dx%d (device=%s)",
        Path(path).name,
        reader.name,
        out.shape[0],
        out.shape[1],
        out.shape[2],
        device,
    )
    return out


def _read_on_gpu(path, backend, device, read_kw):
    """Decode ``path`` on ``device`` (a GPU), returning ``(reader, frames)``.

    A forced ``backend`` is validated and used strictly. ``backend="auto"`` walks
    the installed GPU backends (:func:`~deeperfly.video.base.gpu_reader_candidates`)
    until one actually decodes -- skipping builds whose CUDA support is missing --
    then caches the winner so later reads go straight to it.
    """
    if backend != "auto":
        reader = select_reader(backend, device=device)  # validates install + GPU
        return reader, reader.read(path, device=device, **read_kw)

    candidates = gpu_reader_candidates("auto")
    if not candidates:
        raise RuntimeError(
            f"no GPU video backend installed; install one of {list(GPU_READ_ORDER)}"
        )
    errors = []
    for cls in candidates:
        try:
            frames = cls.read(path, device=device, **read_kw)
        except Exception as exc:
            errors.append(f"{cls.name}: {exc}")
            if len(candidates) > 1:
                # Normal auto-probing (a backend's CUDA build may be missing); the
                # next candidate may work, so keep this quiet unless -v / -vv.
                log.info(
                    "GPU decode via '%s' backend failed (%s); trying next",
                    cls.name,
                    exc,
                )
            continue
        remember_gpu_reader(cls.name)  # probe once: stick with the winner
        return cls, frames
    raise RuntimeError("GPU video decode failed (" + "; ".join(errors) + ")")


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
    device: str = "auto",
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
        ``"auto"`` (the default) and ``"cpu"`` return host NumPy via the parallel
        CPU decode -- already fast, so "auto" does not move image decode to the
        GPU. An explicit CUDA device (``"cuda"`` / ``"cuda:0"``) returns a
        GPU-resident ``torch.Tensor`` (JPEGs via nvJPEG when available), ready for
        :func:`~deeperfly.video.to_jax`.
    indices, start, stop, step
        Select a subset, mirroring :func:`read_video` (``indices`` wins).
    workers
        Decode thread count (default: number of CPUs, capped at the frame count).
    """
    return _read_image_files(
        list_image_files(pattern),
        device=device,
        indices=indices,
        start=start,
        stop=stop,
        step=step,
        workers=workers,
    )


def _read_image_files(
    files: list[Path],
    *,
    device: str = "auto",
    indices: list[int] | None = None,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    workers: int | None = None,
) -> Float[np.ndarray, "T H W 3"]:
    """Decode an already-listed image sequence into ``(T, H, W, 3)`` uint8 RGB.

    The shared tail of :func:`read_images` (which lists/sorts the files) and the
    explicit-file-list path of :func:`read_frames` (which is handed the files a
    recording resolved up front, preserving their natural order). ``files`` is
    sliced by ``indices`` / ``start:stop:step`` exactly as :func:`read_images`.
    """
    device = "cpu" if device == "auto" else canonical_device(device)  # "gpu" -> "cuda"
    files = _subset(list(files), indices, start, stop, step)
    if not files:
        raise ValueError("no frames selected (check indices / start:stop:step)")
    out = (
        _read_images_gpu(files, device, workers)
        if is_gpu_device(device)
        else _read_images_cpu(files, workers)
    )
    log.debug(  # per-read detail (one line per camera per window) -- only at -vv
        "read %d images (imageio) -> %d frames %dx%d (device=%s)",
        len(files),
        out.shape[0],
        out.shape[1],
        out.shape[2],
        device,
    )
    return out


def read_frames(
    source: str | Path | list[Path],
    *,
    backend: str = "auto",
    device: str = "auto",
    indices: list[int] | None = None,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    workers: int | None = None,
) -> Float[np.ndarray, "T H W 3"]:
    """Read frames from a video file **or** an image sequence into ``(T, H, W, 3)``.

    Dispatches on ``source``:

    - a single video file (``.mp4`` / ``.avi`` / ``.mov`` ...) goes to
      :func:`read_video` (``backend`` selects the decoder);
    - a directory or glob of images goes to :func:`read_images` (``workers`` sets
      decode parallelism);
    - an explicit list of footage files -- one video file, or an ordered image
      sequence the caller has already resolved (``deeperfly run`` resolves each
      camera's files up front, naturally sorted) -- is read in the given order
      without re-listing the directory.

    All honor ``device`` (CPU NumPy or GPU tensor) and the same frame selection.
    """
    if isinstance(source, (list, tuple)):
        files = [Path(f) for f in source]
        if not files:
            raise ValueError("read_frames got an empty file list")
        if _is_video_file(files[0]):
            # A camera's video footage is a single file (the resolver keeps just the
            # first when several match), so decode that one.
            return read_video(
                files[0],
                backend=backend,
                device=device,
                start=start,
                stop=stop,
                step=step,
                indices=indices,
            )
        return _read_image_files(
            files,
            device=device,
            indices=indices,
            start=start,
            stop=stop,
            step=step,
            workers=workers,
        )
    p = Path(source)
    if _is_video_file(p):
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
        source,
        device=device,
        indices=indices,
        start=start,
        stop=stop,
        step=step,
        workers=workers,
    )


def reader_name(
    path: str | Path | list[Path], *, backend: str = "auto", device: str = "auto"
) -> str:
    """Name of the decoder :func:`read_frames` would actually use for ``path``.

    Mirrors :func:`read_frames`'s dispatch so logs/diagnostics report the decoder
    that really runs rather than the video backend registry's choice (which only
    applies to video *files*):

    - a video file (``.mp4`` / ``.avi`` ...) resolves through
      :func:`~deeperfly.video.base.select_reader` (``pyav`` / ``opencv`` / a GPU
      backend, honoring ``backend`` and ``device``);
    - an image directory, glob or explicit file list is decoded by
      :func:`read_images`, which ignores the video backend: ``imageio`` on the CPU,
      or ``nvjpeg`` (torchvision) for an all-JPEG set on a CUDA device (it falls
      back to CPU ``imageio`` only if that torchvision build lacks GPU JPEG support).
    """
    if isinstance(path, (list, tuple)):
        files = [Path(f) for f in path]
        if files and _is_video_file(files[0]):
            return select_reader(backend, device=device).name
    elif _is_video_file(Path(path)):
        return select_reader(backend, device=device).name
    else:
        files = list_image_files(path)
    dev = "cpu" if device == "auto" else canonical_device(device)  # match read_images
    if is_gpu_device(dev) and all(f.suffix.lower() in _JPEG_EXTS for f in files):
        return "nvjpeg"
    return "imageio"


def _count_decord(p: Path) -> int:
    import decord

    return len(decord.VideoReader(str(p), ctx=decord.cpu(0)))


def _count_opencv(p: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(p))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _count_pyav(p: Path) -> int:
    import av

    with av.open(str(p)) as container:
        return int(container.streams.video[0].frames)


def _count_imageio(p: Path) -> int:
    # Optional fallback (imageio + bundled ffmpeg); reached only if the core pyav
    # probe and the optional decord/opencv probes above all miss.
    # count_frames_and_secs counts packets via `ffmpeg -c copy` (no pixel decode),
    # so it is exact and fast (~0.1 s here).
    import imageio_ffmpeg

    n, _secs = imageio_ffmpeg.count_frames_and_secs(str(p))
    return int(n)


def count_frames(path: str | Path | list[Path]) -> int | None:
    """Frame count of a video file or image sequence -- ``None`` if unknown.

    Image sequences count their files exactly; videos read it from container
    metadata (PyAV, the core default, always installed -- or decord / OpenCV when
    present) or, failing those, ffmpeg's packet count (the optional imageio
    backend) -- all cheap, with no full pixel decode. This is a **best-effort
    hint** for a progress bar's total: callers
    stream frames and detect end-of-file from the decoder itself, so an off-by-a-few
    count (rare, container-dependent) or a ``None`` never affects correctness.

    Accepts the same sources as :func:`read_frames`, including an explicit list of
    footage files (one video, or an image sequence counted by its length).
    """
    if isinstance(path, (list, tuple)):
        files = [Path(f) for f in path]
        if not files:
            return 0
        if not _is_video_file(files[0]):
            return len(files)  # image sequence: one frame per file
        return count_frames(files[0])  # video footage is a single file
    p = Path(path)
    if not _is_video_file(p):
        try:
            return len(list_image_files(path))
        except FileNotFoundError:
            return None
    for probe in (_count_decord, _count_opencv, _count_pyav, _count_imageio):
        try:
            n = probe(p)
        except Exception:  # backend missing or metadata absent -> try the next
            continue
        if n and n > 0:
            return int(n)
    return None


def _fps_pyav(p: Path) -> float | None:
    import av

    with av.open(str(p)) as container:
        rate = container.streams.video[0].average_rate
    return float(rate) if rate else None


def _fps_opencv(p: Path) -> float | None:
    import cv2

    cap = cv2.VideoCapture(str(p))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return float(fps) if fps and fps > 0 else None


def video_fps(path: str | Path | list[Path]) -> float | None:
    """Frame rate of a video file in frames/sec -- ``None`` if unknown.

    Read from container metadata (PyAV, the core default -- or OpenCV, also core,
    as a fallback); both are cheap and need no pixel decode. Image *sequences*
    carry no intrinsic frame rate, so they return ``None``. Used to detect the
    playback rate of a recording when ``[pipeline].fps`` is left unset and as the
    base rate for the visualization ``speed`` factor. Accepts the same sources as
    :func:`read_frames` (a list resolves to its single video file, else ``None``).
    """
    if isinstance(path, (list, tuple)):
        files = [Path(f) for f in path]
        if files and _is_video_file(files[0]):
            return video_fps(files[0])  # parts share a rate; read it from the first
        return None
    p = Path(path)
    if not _is_video_file(p):
        return None
    for probe in (_fps_pyav, _fps_opencv):
        try:
            fps = probe(p)
        except Exception:  # backend missing or metadata absent -> try the next
            continue
        if fps and fps > 0:
            return float(fps)
    return None


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
    ``[0, 255]``. ``backend`` is ``"auto"`` (pyav, the core default) | ``"pyav"``
    | ``"imageio"`` | ``"opencv"``; ``codec`` overrides the backend default
    (libx264 for pyav/imageio, the ``mp4v`` fourcc for opencv).
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
