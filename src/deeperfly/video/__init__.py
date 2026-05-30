"""Frame I/O and MP4 rendering with pluggable backends.

Reading and writing dispatch over a backend registry so you can choose where
decoding happens and what frames live in:

- CPU readers/writers: ``imageio`` (default), ``opencv``, ``pyav``.
- GPU readers (frames stay a device tensor): ``torchcodec``, ``dali``.

>>> from deeperfly import video
>>> frames = video.read_video("clip.mp4")                       # NumPy, CPU
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
    list_read_backends,
    list_write_backends,
    select_reader,
    select_writer,
    to_jax,
    to_numpy,
)
from .io import read_images, read_video, write_mp4

_LAZY = {"figure_to_array", "render_pose3d_video", "render_overlay_video"}

__all__ = [
    "read_video",
    "read_images",
    "write_mp4",
    "to_numpy",
    "to_jax",
    "select_reader",
    "select_writer",
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
