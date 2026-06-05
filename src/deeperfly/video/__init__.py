"""Frame I/O and MP4 rendering with pluggable backends.

Reading and writing dispatch over a backend registry so you can choose the
decoder/encoder. ``backend="auto"`` (the default) picks the fastest installed
backend -- ``pyav``, the in-process core default, first. All decoding runs on
the CPU.

- Readers: ``pyav`` (default), ``opencv``, ``torchcodec``, ``video_reader_rs``.
- Writers: ``pyav`` (default; in-process H.264), ``opencv``.

>>> from deeperfly import video
>>> frames = video.read_video("clip.mp4")                       # auto: NumPy (host)
>>> frames = video.read_video("clip.mp4", backend="pyav")       # frame-accurate
>>> frames = video.read_video("clip.mp4", backend="torchcodec") # torch tensor (CPU)
>>> video.write_mp4(frames, "out.mp4", fps=30, backend="opencv")
>>> video.available_read_backends()         # varies with installed extras
['opencv', 'pyav']

The rendering helpers (``render_pose3d_video`` / ``render_overlay_video`` /
``figure_to_array``) require the ``viz`` extra and are imported lazily, so plain
read/write does not pull in matplotlib.
"""

from __future__ import annotations

from .base import (
    available_image_readers,
    available_read_backends,
    available_write_backends,
    list_image_readers,
    list_read_backends,
    list_write_backends,
    select_image_reader,
    select_reader,
    select_writer,
    to_numpy,
    to_torch,
)
from .io import (
    count_frames,
    list_image_files,
    read_frames,
    read_images,
    read_video,
    reader_name,
    stream_frames,
    video_fps,
    write_mp4,
)
from .transform import FrameTransform, parse_frame_transforms

_LAZY = {"figure_to_array", "render_pose3d_video", "render_overlay_video"}

__all__ = [
    "read_video",
    "read_images",
    "read_frames",
    "stream_frames",
    "count_frames",
    "video_fps",
    "list_image_files",
    "reader_name",
    "write_mp4",
    "FrameTransform",
    "parse_frame_transforms",
    "to_numpy",
    "to_torch",
    "select_reader",
    "select_writer",
    "select_image_reader",
    "list_read_backends",
    "list_write_backends",
    "list_image_readers",
    "available_read_backends",
    "available_write_backends",
    "available_image_readers",
    "figure_to_array",
    "render_pose3d_video",
    "render_overlay_video",
]


def __getattr__(name: str):  # PEP 562: defer matplotlib-dependent helpers
    if name in _LAZY:
        from . import render

        return getattr(render, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
