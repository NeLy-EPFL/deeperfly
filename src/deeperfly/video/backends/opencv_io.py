"""OpenCV (``cv2``) backend: fast, ubiquitous CPU decode/encode.

OpenCV speaks BGR, so frames are converted to/from RGB at the boundary to keep
the package-wide RGB contract. Random access seeks via ``CAP_PROP_POS_FRAMES``.
"""

from __future__ import annotations

import numpy as np

from ..base import (
    ReaderBackend,
    WriterBackend,
    register_reader,
    register_writer,
)


@register_reader
class OpenCVReader(ReaderBackend):
    name = "opencv"
    requires = ("cv2",)
    supports_seek = True

    @staticmethod
    def _read_sequential(path, start, stop, step):
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"opencv could not open {path!r}")
        out = []
        i = 0
        try:
            while True:
                if stop is not None and i >= stop:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                if i >= start and (i - start) % step == 0:
                    out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                i += 1
        finally:
            cap.release()
        if not out:
            raise ValueError(f"opencv decoded no frames from {path!r}")
        return np.stack(out)

    @classmethod
    def _read_indices(cls, path, indices):
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"opencv could not open {path!r}")
        out = []
        try:
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"opencv could not read frame {i} of {path!r}")
                out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return np.stack(out)


@register_writer
class OpenCVWriter(WriterBackend):
    name = "opencv"
    requires = ("cv2",)

    @staticmethod
    def write(frames, path, *, fps=30.0, codec=None, **kwargs):
        import cv2

        frames = np.asarray(frames)
        # mp4v (MPEG-4) needs even dimensions; trim a stray odd row/column.
        h, w = frames.shape[1] & ~1, frames.shape[2] & ~1
        frames = frames[:, :h, :w]
        fourcc = cv2.VideoWriter_fourcc(*(codec or "mp4v"))
        writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"opencv could not open writer for {path!r}")
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
