"""video_reader-rs backend: a fast Rust/FFmpeg CPU decoder.

``video-reader-rs`` (https://github.com/gcanat/video_reader-rs, imported as
``video_reader``) decodes on multiple threads and exposes native random access
via ``get_batch``. Output is RGB ``(N, H, W, 3)`` NumPy. Both the functional API
and the newer ``PyVideoReader`` class are supported.
"""

from __future__ import annotations

import numpy as np

from ..base import ReaderBackend, register_reader, require_cpu


def _reader(vr, path):
    """A ``PyVideoReader`` instance when available, else ``None`` (functional API)."""
    cls = getattr(vr, "PyVideoReader", None)
    return cls(str(path)) if cls is not None else None


@register_reader
class VideoReaderRsReader(ReaderBackend):
    name = "video_reader_rs"
    requires = ("video_reader",)
    supports_gpu = False
    supports_seek = True

    @staticmethod
    def _read_sequential(path, device, start, stop, step):
        require_cpu(device, "video_reader_rs")
        import video_reader as vr

        reader = _reader(vr, path)
        if reader is not None:
            arr = reader.decode(start_frame=start, end_frame=stop)
        else:
            arr = vr.decode(str(path), start_frame=start, end_frame=stop)
        arr = np.asarray(arr)
        return arr[::step] if step != 1 else arr

    @classmethod
    def _read_indices(cls, path, device, indices):
        require_cpu(device, "video_reader_rs")
        import video_reader as vr

        reader = _reader(vr, path)
        if reader is not None and hasattr(reader, "get_batch"):
            return np.asarray(reader.get_batch(list(indices)))
        if hasattr(vr, "get_batch"):
            return np.asarray(vr.get_batch(str(path), list(indices)))
        return super()._read_indices(path, device, indices)  # decode-and-gather
