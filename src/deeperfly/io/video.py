"""Video-file reading and MP4 writing, backed by PyAV (in-process FFmpeg, CPU).

PyAV is the sole video backend: it links FFmpeg directly and its wheel bundles
FFmpeg, so no system install is needed. :class:`VideoReader` decodes a file to
``(T, H, W, 3)`` uint8 RGB NumPy (frame-accurate, with seeking for random access);
:class:`VideoWriter` encodes frames to H.264 (libx264), one frame, one block, or a
whole array at a time, so a long clip never has to be held in memory at once. ``av``
is imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float

from .base import FrameReader, to_numpy

log = logging.getLogger("deeperfly.io")


class VideoReader(FrameReader):
    """Frame-accurate decode of a single video file via PyAV.

    Sequential reads walk the file forward; indexing with a list seeks per target
    frame (keyframe + decode forward). ``count`` / ``fps`` read container metadata
    -- both cheap, no full pixel decode.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # -- decode (in-process FFmpeg, CPU) -------------------------------------

    def _decode_stream(self, *, start=0, step=1, stop=None):
        """Yield ``(H, W, 3)`` uint8 RGB frames from one forward open-and-walk decode."""
        import av

        with av.open(str(self.path)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if i < start:
                    continue
                if stop is not None and i >= stop:
                    break
                if (i - start) % step == 0:
                    yield frame.to_ndarray(format="rgb24")

    def _decode_range(self, start, stop, step) -> np.ndarray:
        """Decode ``range(start, stop, step)`` to a stacked ``(T, H, W, 3)`` array."""
        out = list(self._decode_stream(start=start, step=step, stop=stop))
        if not out:
            raise ValueError(f"pyav decoded no frames from {str(self.path)!r}")
        return np.stack(out)

    def _decode_indices(self, indices) -> np.ndarray:
        """Random access: seek to the keyframe at/before each target, decode forward to it.

        Recovers each frame's index from its PTS and returns the frames in the order
        ``indices`` requests.
        """
        import av

        picked: dict[int, np.ndarray] = {}
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate
            time_base = stream.time_base
            assert rate is not None and time_base is not None
            for target in sorted(set(indices)):
                # PTS (in time_base units) of the target frame; seek to its keyframe.
                ts = int(target / rate / time_base)
                container.seek(ts, stream=stream, backward=True, any_frame=False)
                for frame in container.decode(stream):
                    assert frame.pts is not None
                    idx = int(round(float(frame.pts * time_base * rate)))
                    if idx >= target:
                        picked[target] = frame.to_ndarray(format="rgb24")
                        break
        try:
            return np.stack([picked[int(i)] for i in indices])
        except KeyError as exc:  # a seek overshot / frame missing
            raise ValueError(
                f"pyav could not seek to frame {exc} of {str(self.path)!r}"
            ) from None

    def __getitem__(self, key: int | list[int] | slice) -> Float[np.ndarray, "..."]:
        if isinstance(key, int):
            out = self._decode_range(key, key + 1, 1)[0]
        elif isinstance(key, list):
            idx = [int(i) for i in key]
            if not idx:
                raise ValueError("index list must be non-empty")
            out = self._decode_indices(idx)
        elif isinstance(key, slice):
            start, stop, step = key.start or 0, key.stop, key.step or 1
            out = self._decode_range(int(start), stop, int(step))
        else:
            raise TypeError(f"invalid index type {type(key).__name__!r}")
        log.debug(
            "read video %s via pyav -> %s",
            self.path.name,
            out.shape,
        )
        return out

    def stream_frames(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Iterator[Float[np.ndarray, "H W 3"]]:
        yield from self._decode_stream(start=start, stop=stop, step=step)

    def stream_blocks(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
        block_size: int = 64,
    ) -> Iterator[Float[np.ndarray, "T H W 3"]]:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        buf: list[np.ndarray] = []
        for frame in self._decode_stream(start=start, stop=stop, step=step):
            buf.append(frame)
            if len(buf) >= block_size:
                yield np.stack(buf)
                buf = []
        if buf:
            yield np.stack(buf)

    # -- metadata probes (container, no pixel decode) ------------------------

    def count(self) -> int | None:
        """Frame count from the container header, or ``None`` if it is absent.

        Some containers (raw / transport streams, some MKV) omit ``nb_frames``;
        an exact count then needs a full decode, so this returns ``None`` rather
        than a ``duration * fps`` estimate.
        """
        import av

        try:
            with av.open(str(self.path)) as container:
                n = container.streams.video[0].frames
        except Exception:  # unreadable / unsupported container -> unknown
            return None
        return int(n) if n and n > 0 else None

    def fps(self) -> float | None:
        """Average frame rate from the container header, or ``None`` if unavailable."""
        import av

        try:
            with av.open(str(self.path)) as container:
                stream = container.streams.video[0]
                rate = (
                    stream.average_rate or stream.guessed_rate
                )  # match _decode_indices
        except Exception:  # unreadable / unsupported container -> unknown
            return None
        return float(rate) if rate else None


class VideoWriter:
    """Incremental H.264 (libx264) MP4 encoder, backed by PyAV.

    Open it, feed frames, close it (or use it as a context manager).
    :meth:`write_frame` appends one ``(H, W, 3)`` frame; :meth:`write_frames`
    appends a whole ``(T, H, W, 3)`` array or any iterable of frames / blocks -- so
    a long clip can be encoded as it is produced, without ever holding every frame
    in memory:

    >>> with VideoWriter("out.mp4", fps=30) as writer:
    ...     for frame in render():          # a (H, W, 3) frame
    ...         writer.write_frame(frame)

    The container and stream are opened lazily on the first frame (its size sets the
    encode dimensions, rounded down to even for ``yuv420p`` subsampling); later
    frames are cropped to match. Non-``uint8`` input is clipped to ``[0, 255]``.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float = 30.0,
        *,
        codec: str | None = None,
        pix_fmt: str = "yuv420p",
    ) -> None:
        self.path = Path(path)
        self.fps = fps
        self.codec = codec
        self.pix_fmt = pix_fmt
        self._container: Any = None  # av.container.OutputContainer (lazy, on 1st frame)
        self._stream: Any = None  # av.video.stream.VideoStream
        self._size: tuple[int, int] | None = None  # (w, h), even

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _open(self, width: int, height: int) -> None:
        import av

        w, h = width & ~1, height & ~1  # yuv420p subsampling needs even dimensions
        rate = Fraction(self.fps).limit_denominator(1_000_000)
        self._container = av.open(str(self.path), mode="w")
        stream = self._container.add_stream(self.codec or "libx264", rate=rate)
        stream.width = w
        stream.height = h
        stream.pix_fmt = self.pix_fmt
        self._stream = stream
        self._size = (w, h)
        log.info("writing %s via pyav: %dx%d @ %g fps", self.path.name, w, h, self.fps)

    def write_frame(self, frame) -> None:
        """Append a single ``(H, W, 3)`` frame (non-``uint8`` is clipped to ``[0, 255]``).

        The first frame's size sets the encode dimensions (rounded down to even for
        ``yuv420p`` subsampling); later frames are cropped to match.

        Parameters
        ----------
        frame
            One ``(H, W, 3)`` RGB frame (NumPy, or a torch / DLPack array).
        """
        import av

        frame = to_numpy(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if self._stream is None:
            self._open(frame.shape[1], frame.shape[0])
        assert self._size is not None
        w, h = self._size
        vframe = av.VideoFrame.from_ndarray(
            np.ascontiguousarray(frame[:h, :w]), format="rgb24"
        )
        for packet in self._stream.encode(vframe):
            self._container.mux(packet)

    def write_frames(self, frames) -> None:
        """Append many frames: a ``(T, H, W, 3)`` batch, or any iterable of frames.

        Accepts a NumPy array (each frame along axis 0), a torch / DLPack batch, or
        any iterable of frames or blocks (e.g. a generator) -- so frames can be
        encoded as they arrive, without holding the whole clip in memory.

        Parameters
        ----------
        frames
            A batch, or an iterable of frames / batches (non-``uint8`` is clipped).
        """
        if isinstance(frames, np.ndarray):
            if frames.ndim == 4:
                for frame in frames:
                    self.write_frame(frame)
            else:
                self.write_frame(frames)
            return
        if hasattr(frames, "detach") or hasattr(frames, "__dlpack__"):  # torch/array
            self.write_frames(to_numpy(frames))
            return
        for item in frames:  # a list / tuple / generator of frames (or batches)
            self.write_frames(item)

    def close(self) -> None:
        """Flush the encoder and close the file (idempotent)."""
        if self._container is None:
            return
        try:
            for packet in self._stream.encode():  # flush
                self._container.mux(packet)
        finally:
            self._container.close()
            self._container = None
            self._stream = None
