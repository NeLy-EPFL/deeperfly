"""Frame I/O and MP4 rendering with pluggable backends.

Reading and writing dispatch over a backend registry so you can choose where
decoding happens and what frames live in. ``backend="auto"`` and ``device="auto"``
(both defaults) pick the fastest available path: GPU/NVDEC decode when a GPU and a
GPU backend are present, otherwise the fastest installed CPU decoder (imageio,
which forks an ``ffmpeg`` subprocess, is the last resort).

- CPU readers: ``decord``, ``video_reader_rs``, ``torchcodec``, ``pyav``,
  ``opencv``, ``imageio``.
- GPU readers (frames stay a device tensor): ``torchcodec``, ``decord``,
  ``dali``.
- Writers: ``pyav`` (default; in-process H.264), ``imageio``, ``opencv``.

>>> from deeperfly import video
>>> frames = video.read_video("clip.mp4")                       # auto: NumPy (host)
>>> frames = video.read_video("clip.mp4", backend="pyav")       # frame-accurate
>>> frames = video.read_video("clip.mp4", backend="torchcodec", device="cuda")
>>> video.write_mp4(frames, "out.mp4", fps=30, backend="opencv")
>>> video.available_read_backends()
['imageio', 'opencv']

The rendering helpers (``render_pose3d_video`` / ``render_overlay_video`` /
``figure_to_array``) require the ``viz`` extra and are imported lazily, so plain
read/write does not pull in matplotlib.
"""

from __future__ import annotations

from .base import (
    available_read_backends,
    available_write_backends,
    cuda_available,
    list_read_backends,
    list_write_backends,
    resolve_device,
    select_reader,
    select_writer,
    to_jax,
    to_numpy,
)
from .io import (
    count_frames,
    list_image_files,
    read_frames,
    read_images,
    read_video,
    write_mp4,
)

_LAZY = {"figure_to_array", "render_pose3d_video", "render_overlay_video"}

__all__ = [
    "read_video",
    "read_images",
    "read_frames",
    "count_frames",
    "list_image_files",
    "write_mp4",
    "to_numpy",
    "to_jax",
    "select_reader",
    "select_writer",
    "resolve_device",
    "cuda_available",
    "list_read_backends",
    "list_write_backends",
    "available_read_backends",
    "available_write_backends",
    "figure_to_array",
    "render_pose3d_video",
    "render_overlay_video",
]


def __getattr__(name: str):  # PEP 562: defer matplotlib-dependent helpers
    if name in _LAZY:
        from . import render

        return getattr(render, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
