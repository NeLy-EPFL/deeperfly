"""PyAV (``av``) backend: direct FFmpeg bindings, frame-accurate decode/encode.

Random access seeks to the keyframe at or before each target (``container.seek``
with ``backward=True``) then decodes forward to the exact frame, recovering the
frame index from its PTS.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..base import (
    ReaderBackend,
    WriterBackend,
    register_reader,
    register_writer,
)


@register_reader
class PyAVReader(ReaderBackend):
    name = "pyav"
    requires = ("av",)
    supports_seek = True

    @staticmethod
    def stream(path, *, start=0, step=1, stop=None):
        """Single open-and-walk decode: one pass for the whole stream (no re-open).

        ``stop`` is an internal bound used by :meth:`_read_sequential`; the public
        :class:`~deeperfly.video.base.ReaderBackend` streaming contract is
        open-ended (decode to end-of-stream), so callers omit it.
        """
        import av

        with av.open(str(path)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i < start:
                    continue
                if stop is not None and i >= stop:
                    break
                if (i - start) % step == 0:
                    yield frame.to_ndarray(format="rgb24")

    @classmethod
    def _read_sequential(cls, path, start, stop, step):
        out = list(cls.stream(path, start=start, step=step, stop=stop))
        if not out:
            raise ValueError(f"pyav decoded no frames from {path!r}")
        return np.stack(out)

    @classmethod
    def _read_indices(cls, path, indices):
        import av

        picked: dict[int, np.ndarray] = {}
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate
            time_base = stream.time_base
            for target in sorted(set(indices)):
                # PTS (in time_base units) of the target frame, seek to its keyframe.
                ts = int(target / rate / time_base)
                container.seek(ts, stream=stream, backward=True, any_frame=False)
                for frame in container.decode(stream):
                    idx = int(round(float(frame.pts * time_base * rate)))
                    if idx >= target:
                        picked[target] = frame.to_ndarray(format="rgb24")
                        break
        try:
            return np.stack([picked[int(i)] for i in indices])
        except KeyError as exc:  # a seek overshot / frame missing
            raise ValueError(
                f"pyav could not seek to frame {exc} of {path!r}"
            ) from None


@register_writer
class PyAVWriter(WriterBackend):
    name = "pyav"
    requires = ("av",)

    @staticmethod
    def write(frames, path, *, fps=30.0, codec=None, pix_fmt="yuv420p", **kwargs):
        import av

        frames = np.asarray(frames)
        # yuv420p subsampling needs even dimensions.
        h, w = frames.shape[1] & ~1, frames.shape[2] & ~1
        frames = frames[:, :h, :w]
        rate = Fraction(fps).limit_denominator(1_000_000)
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream(codec or "libx264", rate=rate)
            stream.width = w
            stream.height = h
            stream.pix_fmt = pix_fmt
            for frame in frames:
                vframe = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(frame), format="rgb24"
                )
                for packet in stream.encode(vframe):
                    container.mux(packet)
            for packet in stream.encode():  # flush
                container.mux(packet)
