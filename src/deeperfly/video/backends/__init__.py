"""Importing this package registers every built-in video backend.

Each module is import-cheap: the heavy third-party libraries (cv2, av,
torchcodec, decord, video_reader, PyNvVideoCodec, nvidia.dali, imageio) are
imported lazily inside the backends, so importing all of them here only
populates the registry.
"""

from __future__ import annotations

from . import (  # noqa: F401  (imported for registration side effects)
    dali_io,
    decord_io,
    imageio_io,
    opencv_io,
    pyav_io,
    pynvvideocodec_io,
    torchcodec_io,
    video_reader_rs_io,
)

__all__ = [
    "imageio_io",
    "opencv_io",
    "pyav_io",
    "decord_io",
    "video_reader_rs_io",
    "torchcodec_io",
    "pynvvideocodec_io",
    "dali_io",
]
