"""Importing this package registers every built-in video backend.

Each module is import-cheap: the heavy third-party libraries (cv2, av,
torchcodec, video_reader) are imported lazily inside the backends, so importing
all of them here only populates the registry.
"""

from __future__ import annotations

from . import (  # noqa: F401  (imported for registration side effects)
    opencv_io,
    pyav_io,
    torchcodec_io,
    video_reader_rs_io,
)

__all__ = [
    "opencv_io",
    "pyav_io",
    "video_reader_rs_io",
    "torchcodec_io",
]
