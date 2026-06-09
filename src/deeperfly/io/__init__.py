"""Frame I/O: reader classes for video files and image sequences, plus MP4 writing.

Footage is read through a small class hierarchy rooted at
:class:`~deeperfly.io.base.FrameReader`:

- :class:`~deeperfly.io.video.VideoReader` -- frame-accurate decode of a video file
  to ``(T, H, W, 3)`` uint8 RGB NumPy (PyAV, in-process FFmpeg, CPU).
- :class:`~deeperfly.io.images.ImageSequenceReader` -- parallel decode of an image
  sequence (OpenCV).

:func:`open_reader` resolves a source (video file, image directory/glob, or explicit
footage file list) to the right reader **once**; callers then index (``reader[:]``,
``reader[i]``, ``reader[[0,3,5]]``) or stream (``stream_frames`` / ``stream_blocks``)
against that object. :class:`~deeperfly.io.video.VideoWriter` encodes frames to H.264,
one frame or one array at a time.

>>> from deeperfly import io
>>> reader = io.open_reader("clip.mp4")
>>> frames = reader[:]                     # (T, H, W, 3) uint8, host NumPy
>>> with io.VideoWriter("out.mp4", fps=30) as writer:
...     writer.write_frames(frames)            # a batch or iterable; or write_frame()

Pose overlays and 3D reconstructions are rendered to MP4 by
:mod:`deeperfly.visualization.compose` (the OpenCV panel compositor), which builds
on these read/write primitives.
"""

from __future__ import annotations

from pathlib import Path

from .base import (
    IMAGE_EXTS,
    VIDEO_EXTS,
    FrameReader,
    is_video_file,
    to_numpy,
    to_torch,
)
from .images import (
    IMAGE_READ_ORDER,
    ImageSequenceReader,
    available_image_readers,
    list_image_files,
    list_image_readers,
    select_image_reader,
)
from .video import VideoReader, VideoWriter


def open_reader(
    source: str | Path | list[Path],
    *,
    workers: int | None = None,
) -> FrameReader:
    """Open the right :class:`FrameReader` for a footage source.

    Dispatches on ``source`` (the one place this dispatch lives):

    - a single video file (``.mp4`` / ``.avi`` / ``.mov`` ...) -> a
      :class:`~deeperfly.io.video.VideoReader` (PyAV);
    - a directory or glob of images -> an
      :class:`~deeperfly.io.images.ImageSequenceReader` (OpenCV, ``workers`` sets
      decode parallelism);
    - an explicit list of footage files -- one video file, or an ordered image
      sequence the caller has already resolved (``deeperfly run`` resolves each
      camera's files up front, naturally sorted) -- is read in the given order
      without re-listing the directory.

    Parameters
    ----------
    source
        A video file, an image directory/glob, or an explicit list of footage
        files (one video, or an ordered image sequence).
    workers
        Decode thread count for image sequences.

    Returns
    -------
    FrameReader
        A :class:`~deeperfly.io.video.VideoReader` or
        :class:`~deeperfly.io.images.ImageSequenceReader`.

    Raises
    ------
    ValueError
        If an explicit file list is empty.
    """
    if isinstance(source, (list, tuple)):
        files = [Path(f) for f in source]
        if not files:
            raise ValueError("open_reader got an empty file list")
        # A camera's video footage is a single file (the resolver keeps just the
        # first when several match), so decode that one.
        if is_video_file(files[0]):
            return VideoReader(files[0])
        return ImageSequenceReader(files, workers=workers)
    p = Path(source)
    if is_video_file(p):
        return VideoReader(p)
    return ImageSequenceReader.from_pattern(source, workers=workers)


__all__ = [
    "FrameReader",
    "VideoReader",
    "VideoWriter",
    "ImageSequenceReader",
    "open_reader",
    "to_numpy",
    "to_torch",
    "select_image_reader",
    "list_image_readers",
    "available_image_readers",
    "list_image_files",
    "is_video_file",
    "IMAGE_READ_ORDER",
    "VIDEO_EXTS",
    "IMAGE_EXTS",
]
