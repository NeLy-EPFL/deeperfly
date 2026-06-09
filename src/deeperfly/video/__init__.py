"""Frame I/O and MP4 writing, backed by PyAV.

Reading and writing dispatch over a small backend registry; ``backend="auto"``
(the default) resolves to ``pyav``, the sole video backend -- in-process FFmpeg
(its wheel bundles FFmpeg), decoding and encoding H.264 on the CPU.

- Reader: ``pyav`` -- frame-accurate decode to ``(T, H, W, 3)`` uint8 NumPy.
- Writer: ``pyav`` -- H.264 (libx264) encode.

>>> from deeperfly import video
>>> frames = video.read_video("clip.mp4")           # (T, H, W, 3) uint8, host NumPy
>>> video.write_mp4(frames, "out.mp4", fps=30)

Pose overlays and 3D reconstructions are rendered to MP4 by
:mod:`deeperfly.viz.compose` (the OpenCV panel compositor), which builds on these
read/write primitives.
"""

from __future__ import annotations

from .base import (
    available_image_readers,
    list_image_readers,
    select_image_reader,
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
    "select_image_reader",
    "list_image_readers",
    "available_image_readers",
]
