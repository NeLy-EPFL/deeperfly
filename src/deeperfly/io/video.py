"""Video-file reading and MP4 writing, backed by PyAV (in-process FFmpeg, CPU).

PyAV is the sole video backend: it links FFmpeg directly and its wheel bundles
FFmpeg, so no system install is needed. :class:`VideoReader` decodes a file to
``(T, H, W, 3)`` uint8 RGB NumPy (frame-accurate, with seeking for random access);
:class:`VideoCursor` is its stateful counterpart for a caller reading single frames in
an unpredictable order (a viewer), keeping the container and decoder open between reads;
:class:`VideoWriter` encodes frames to H.264 (libx264), one frame, one block, or a
whole array at a time, so a long clip never has to be held in memory at once. ``av``
is imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float

from .base import FrameCursor, FrameReader, to_numpy

log = logging.getLogger("deeperfly.io")

# The neutral chroma value: in planar YUV, a picture with no color has both chroma
# planes pinned here.
_NEUTRAL_CHROMA = 128


def _cursor_thread_count() -> int:
    """Decode threads per stream for a cursor: a fraction of the host's cores.

    ``thread_type = "AUTO"`` alone lets FFmpeg size the pool from the core count, which is
    right for one stream and wrong for a rig: each of eight cameras then spawns a pool that
    large and the eight concurrent decodes of one navigation oversubscribe the machine.
    Measured on a 32-core host, eight cameras served at once, median wall time for a jump
    (which is a walk forward from a keyframe, the work that dominates):

    ==================  ========
    threads per stream  jump
    ==================  ========
    uncapped (AUTO)     395 ms
    2                   358 ms
    8                   192 ms
    16                  201 ms
    ==================  ========

    So there is a broad optimum near ``cores / 4`` -- enough frame-threading depth to keep
    a single walk moving, few enough that eight of them do not thrash -- and it degrades in
    both directions. A floor of 2 keeps a small host from serializing each walk, which is
    the worst case of all (one thread measured 213 ms for a *single* camera's back-step).
    """
    return max(2, (os.cpu_count() or 4) // 4)


def _carries_no_color(frame) -> bool:
    """Whether a planar-YUV frame's chroma is flat, i.e. the picture is really gray.

    Monochrome footage -- every fly rig here -- is still stored as planar YUV with both
    chroma planes pinned at :data:`_NEUTRAL_CHROMA`. Spotting that lets a caller take the
    frame as one channel instead of three, which skips the YUV->RGB conversion, skips the
    channel-reversing copy an encoder wants, and cuts the JPEG it produces to a third of
    the CPU.

    Decided per **frame** and never remembered per file: a color video whose first frame
    happens to be black or uniformly gray has flat chroma *there* and color everywhere
    after it, so a cached verdict would serve the rest of that video stripped of its
    color. The test costs ~0.025 ms on a 1984x512 frame, far less than the copy it saves.

    Returns ``False`` for anything that is not 3-plane planar YUV (packed RGB, NV12,
    already-gray), which is the conservative answer: the caller then decodes as usual.
    """
    planes = list(frame.planes)
    if len(planes) < 3:
        return False
    for plane in planes[1:3]:
        # A plane's buffer is padded out to `line_size`; only the first `width` bytes of
        # each row are picture and the padding holds arbitrary values, so testing the raw
        # buffer would report color that isn't there.
        data = np.frombuffer(plane, dtype=np.uint8)
        if data.size < plane.height * plane.line_size or plane.width == 0:
            return False
        rows = data.reshape(plane.height, plane.line_size)[:, : plane.width]
        if rows.min() != _NEUTRAL_CHROMA or rows.max() != _NEUTRAL_CHROMA:
            return False
    return True


#: ``AVCOL_RANGE_JPEG`` -- luma spans the full 0..255 rather than the 16..235 of TV range.
_COLOR_RANGE_FULL = 2


def _is_full_range(frame) -> bool:
    """Whether the frame's luma is FULL range, so that ``Y`` is already the gray picture.

    This is the condition under which taking the luma plane is not merely cheaper but
    *identical*. In full range (``yuvj*`` / ``AVCOL_RANGE_JPEG``, what these cameras record)
    a color-free frame has ``R = G = B = Y`` exactly, so the plane is the same picture the
    YUV->RGB conversion would build. In TV range it is not: the conversion expands 16..235
    to 0..255, and the plane is then a different set of numbers from the RGB the rest of the
    pipeline is calibrated on -- measured 33 against 19 on the same pixel. So TV-range
    footage is decoded as RGB even when it carries no color, and pays for the conversion.
    """
    fmt = frame.format.name
    if fmt.startswith("yuvj") or fmt.startswith("gray"):
        return True  # both are full range by definition
    return int(getattr(frame, "color_range", 0) or 0) == _COLOR_RANGE_FULL


def _luma_plane(frame) -> np.ndarray | None:
    """The frame's ``(H, W)`` luma plane as a NumPy copy, or ``None`` if it has none.

    For planar YUV -- what every camera here records -- the luma plane *is* the gray
    picture, already in the decoded frame. Taking it is a memcpy of one plane;
    ``to_ndarray(format="gray")`` instead runs the frame through swscale, which on this
    footage costs 20-30 ms a frame against 0.1-0.5 ms here (measured on 1600x1008 and
    960x512 h264, 100-300x) -- and the conversion, not the H.264 decode, is most of what
    reading a frame costs.

    Callers must have established :func:`_is_full_range` first: the plane is only the same
    picture as the RGB decode when the luma is not range-scaled.

    Each row's buffer is padded out to ``line_size``; only the first ``width`` bytes are
    picture, so the padding is sliced off (and the copy that ``ascontiguousarray`` makes is
    what lets the frame be released).
    """
    fmt = frame.format.name
    if not (fmt.startswith("yuv") or fmt.startswith("gray")):
        return None
    plane = frame.planes[0]
    if plane.width != frame.width or plane.height != frame.height:
        return None  # subsampled or unexpected layout -- let swscale answer
    data = np.frombuffer(plane, dtype=np.uint8)
    if data.size < plane.height * plane.line_size:
        return None
    rows = data.reshape(plane.height, plane.line_size)[:, : plane.width]
    return np.ascontiguousarray(rows)


def _stack_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Stack a block's frames, promoting one channel to three if the block is mixed.

    ``gray_ok`` is decided per **frame** (a color video's black opening frame carries no
    color *there*), so a block can in principle hold both kinds. Rather than refuse the
    stack, the odd one out is broadcast to three channels -- which is what it means.
    """
    widths = {f.shape[-1] for f in frames}
    if len(widths) > 1:
        frames = [np.repeat(f, 3, axis=-1) if f.shape[-1] == 1 else f for f in frames]
    return np.stack(frames)


def _frame_array(
    frame,
    *,
    gray_ok: bool,
    keep_channel_axis: bool = False,
    same_as_rgb: bool = False,
) -> np.ndarray:
    """A decoded PyAV frame as ``(H, W, 3)`` RGB, or one channel when it has no color.

    Only ever collapses to one channel when the caller passed ``gray_ok``, and then only for
    a frame that carries no chroma (:func:`_carries_no_color`).

    ``same_as_rgb`` additionally requires the gray to be *numerically* what the RGB decode
    would have produced, which needs the luma to be full range as well
    (:func:`_is_full_range`) -- in TV range the conversion expands 16..235 to 0..255 and the
    plane is a different set of numbers. Two callers, two needs: a viewer wants a cheap
    picture to LOOK at, so a range-scaled gray is still the right picture; a detector's
    frames are arithmetic, and one that used to see the RGB luma must keep seeing it.

    ``(H, W)`` is what a cursor's callers expect; ``keep_channel_axis`` returns ``(H, W, 1)``
    instead, which is what the batch paths want -- every frame op and every model indexes the
    channel axis from the right (``[..., :3]``, ``shape[-1]``), so a kept axis of length 1
    flows through them unchanged while a 2-D frame would silently crop the wrong axes.
    """
    if not gray_ok:
        return frame.to_ndarray(format="rgb24")
    if same_as_rgb and not _is_full_range(frame):
        return frame.to_ndarray(format="rgb24")
    if frame.format.name.startswith("gray") or _carries_no_color(frame):
        gray = _luma_plane(frame) if _is_full_range(frame) else None
        if gray is None:
            gray = frame.to_ndarray(format="gray")
        return gray[..., None] if keep_channel_axis else gray
    return frame.to_ndarray(format="rgb24")


class VideoReader(FrameReader):
    """Frame-accurate decode of a single video file via PyAV.

    Sequential reads walk the file forward; indexing with a single index or a list
    seeks per target frame (keyframe + decode forward). ``count`` / ``fps`` read
    container metadata -- both cheap, no full pixel decode.

    Every method here opens and closes the file, which keeps the reader itself cheap to
    hold and safe to hand between processes. A caller reading single frames repeatedly
    should take a :class:`VideoCursor` from :meth:`cursor` instead, which keeps the
    container open across reads.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # -- decode (in-process FFmpeg, CPU) -------------------------------------

    def _decode_stream(
        self, *, start=0, step=1, stop=None, gray_ok=False, thread_count=None
    ):
        """Yield uint8 frames from one forward open-and-walk decode.

        ``(H, W, 3)`` RGB, or ``(H, W, 1)`` luma when ``gray_ok`` and the picture carries
        no color (see :func:`_frame_array`).

        The video stream is decoded with ``thread_type = "AUTO"`` (FFmpeg
        frame/slice multithreading), which is several times faster than the
        single-threaded default on multi-core hosts. ``thread_count`` caps the pool:
        ``AUTO`` alone sizes it from the core count, which is right for one stream and
        oversubscribes the host when a rig's eight cameras each open one (see
        :func:`_cursor_thread_count`). ``None`` leaves FFmpeg's choice.
        """
        import av

        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            if thread_count:
                # Before the first decode: the count is read when the codec opens.
                stream.codec_context.thread_count = int(thread_count)
            for i, frame in enumerate(container.decode(stream)):
                if i < start:
                    continue
                if stop is not None and i >= stop:
                    break
                if (i - start) % step == 0:
                    yield _frame_array(
                        frame,
                        gray_ok=gray_ok,
                        keep_channel_axis=True,
                        # A batch read feeds detectors and calibration, so gray is taken
                        # only where it is the same numbers the RGB decode would give.
                        same_as_rgb=True,
                    )

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
            stream.thread_type = "AUTO"
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

    def _decode_one(self, idx: int) -> np.ndarray:
        """One frame, by seeking to its keyframe, falling back to a forward walk.

        The walk (:meth:`_decode_range`) decodes every frame from the start of the file
        and throws away the ones before ``idx`` -- ``continue`` in the consumer loop skips
        the array conversion, not the decode -- so it costs O(``idx``), growing with the
        index, which is why a viewer of a long recording got slower the further in the
        operator worked. Seeking costs O(GOP) instead: for a viewer reading eight cameras
        of 3000-frame footage, a median navigation went from 2664 ms to 245 ms.

        The walk stays as the fallback, for a container that cannot seek or carries no
        timestamps, and for a negative index -- unsupported either way, but this keeps the
        error the one callers have always seen. The two paths were verified byte-identical
        across recordings, cameras and indices, including frame 0, both sides of a GOP
        boundary, and the last frame.
        """
        if idx >= 0:
            try:
                return self._decode_indices([idx])[0]
            except Exception as exc:  # noqa: BLE001 -- unseekable container / no PTS
                log.debug(
                    "could not seek to frame %d of %s (%s); walking instead",
                    idx,
                    self.path.name,
                    exc,
                )
        return self._decode_range(idx, idx + 1, 1)[0]

    def __getitem__(self, key: int | list[int] | slice) -> Float[np.ndarray, "..."]:
        if isinstance(key, int):
            out = self._decode_one(int(key))
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
        gray_ok: bool = False,
        thread_count: int | None = None,
    ) -> Iterator[Float[np.ndarray, "H W 3"]]:
        yield from self._decode_stream(
            start=start,
            stop=stop,
            step=step,
            gray_ok=gray_ok,
            thread_count=thread_count,
        )

    def stream_blocks(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
        block_size: int = 64,
        gray_ok: bool = False,
        thread_count: int | None = None,
    ) -> Iterator[Float[np.ndarray, "T H W 3"]]:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        buf: list[np.ndarray] = []
        for frame in self._decode_stream(
            start=start,
            stop=stop,
            step=step,
            gray_ok=gray_ok,
            thread_count=thread_count,
        ):
            buf.append(frame)
            if len(buf) >= block_size:
                yield _stack_frames(buf)
                buf = []
        if buf:
            yield _stack_frames(buf)

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

    def cursor(
        self, *, gray_ok: bool = False, same_as_rgb: bool = False
    ) -> FrameCursor:
        """A :class:`VideoCursor` over this file, with the decoder held open.

        ``same_as_rgb`` narrows ``gray_ok`` to frames whose gray is numerically what the RGB
        decode would give -- what a caller whose frames are arithmetic rather than a picture
        needs (see :func:`_frame_array`).

        Falls back to the stateless base cursor if the container cannot be opened or
        carries no usable timestamps -- so a caller always gets something that works.
        """
        try:
            return VideoCursor(self.path, gray_ok=gray_ok, same_as_rgb=same_as_rgb)
        except Exception as exc:  # noqa: BLE001 -- unreadable / no PTS mapping
            log.debug(
                "no persistent cursor for %s (%s); using stateless reads",
                self.path.name,
                exc,
            )
            return FrameCursor(self)


class VideoCursor(FrameCursor):
    """Random access to one video file with the decoder held open between reads.

    :meth:`VideoReader.__getitem__` answers each frame from a freshly opened container: it
    re-parses the header, seeks, decodes one picture and throws the decoder away. A viewer
    reads one frame per camera at a time, so it pays that setup on every frame -- and
    worse, it re-walks from the keyframe every time, even when the frame it wants is the
    one the decoder was about to produce anyway. This class keeps the container, the stream
    and the live decode generator, so:

    - stepping to the **next** frame is one ``next()`` on that generator -- no seek, no
      open, no re-walk;
    - a jump anywhere else is a seek on a container that is already open.

    Measured on 1984x512 100 fps footage, all eight cameras served at once, decode plus
    JPEG encode, median over jumps spread across the GOP:

    ==========================  ========  =============
    per navigation              jump      step forward
    ==========================  ========  =============
    walk from frame 0           2664 ms   2224 ms
    seek, re-open every read     245 ms    452 ms
    cursor held open             195 ms      5 ms
    ==========================  ========  =============

    A jump still has to decode forward from a keyframe -- some 125 frames on average here,
    which is the whole of that remaining 195 ms and is a property of how the footage was
    encoded, not of this code. A step forward decodes exactly one frame.

    An ``av`` container cannot be decoded from two threads at once, so every read takes
    :attr:`_lock`. One cursor per camera means the eight views of a navigation still decode
    in parallel; two requests for the *same* camera serialize, which is what a prefetch
    landing on top of a navigation does.

    Parameters
    ----------
    path
        The video file.
    gray_ok
        Whether single-channel frames may be returned for pictures with no color
        (see :meth:`FrameReader.cursor`).
    thread_count
        Decode threads for this stream; ``None`` takes :func:`_cursor_thread_count` (which
        explains why a viewer caps it at all) and ``0`` leaves FFmpeg's own choice.

    Raises
    ------
    ValueError
        If the stream carries no frame rate or time base, so frame indices cannot be
        mapped to timestamps to seek with.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        gray_ok: bool = False,
        thread_count: int | None = None,
        same_as_rgb: bool = False,
    ) -> None:
        import av

        if thread_count is None:
            thread_count = _cursor_thread_count()
        self.path = Path(path)
        self._gray_ok = gray_ok
        self._same_as_rgb = same_as_rgb
        # Held for the whole of every read, so `close` can never free the container out
        # from under a decode in flight -- it waits for the reader to leave instead.
        self._lock = threading.Lock()
        self._container: Any = av.open(str(self.path))
        try:
            stream = self._container.streams.video[0]
            stream.thread_type = "AUTO"
            if thread_count:
                # Must happen before the first decode, while the codec is still unopened.
                try:
                    stream.codec_context.thread_count = thread_count
                except Exception as exc:  # noqa: BLE001 -- some codecs refuse
                    log.debug("could not cap decode threads for %s: %s", path, exc)
            rate = stream.average_rate or stream.guessed_rate
            if not rate or stream.time_base is None:
                raise ValueError(f"{self.path.name!r} has no frame rate to index by")
            self._stream = stream
            self._rate = rate
            self._time_base = stream.time_base
        except Exception:
            self._container.close()
            self._container = None
            raise
        # The live decode generator and the index it will yield next -- the pair that
        # makes a step forward free. `None` / `-1` means "nothing to continue from".
        self._gen: Iterator[Any] | None = None
        self._next = -1

    def _index_of(self, frame) -> int:
        """The frame's own index, recovered from its presentation timestamp."""
        if frame.pts is None:
            raise ValueError(f"{self.path.name!r} has frames with no timestamp")
        return int(round(float(frame.pts * self._time_base * self._rate)))

    def _seek_to(self, idx: int):
        """Seek to ``idx``'s keyframe and decode forward to it, leaving the gen live."""
        if idx < 0:
            raise IndexError(f"frame index {idx} is negative")
        self._gen = None
        ts = int(idx / self._rate / self._time_base)
        self._container.seek(ts, stream=self._stream, backward=True, any_frame=False)
        gen = self._container.decode(self._stream)
        for frame in gen:
            got = self._index_of(frame)
            if got >= idx:
                # Keep the generator: the next frame is very often the one asked for next.
                self._gen, self._next = gen, got + 1
                return frame
        raise ValueError(f"could not seek to frame {idx} of {str(self.path)!r}")

    def frame(self, idx: int) -> np.ndarray:
        with self._lock:
            if self._container is None:
                raise ValueError(f"cursor over {self.path.name!r} is closed")
            if self._gen is not None and self._next == idx:
                frame = self._continue(idx)
                if frame is not None:
                    return _frame_array(
                        frame,
                        gray_ok=self._gray_ok,
                        same_as_rgb=self._same_as_rgb,
                    )
            return _frame_array(
                self._seek_to(idx),
                gray_ok=self._gray_ok,
                same_as_rgb=self._same_as_rgb,
            )

    def _continue(self, idx: int):
        """The next frame off the live generator if it really is ``idx``, else ``None``.

        ``None`` means the caller should seek: the stream ended, or the decoder handed
        back a different index than the walk predicted (a dropped or repeated frame). The
        index is re-derived from the timestamp rather than counted, so a stream that skips
        can never make this hand back a picture from the wrong moment.
        """
        try:
            frame = next(self._gen)  # type: ignore[arg-type]
        except StopIteration:
            self._gen = None
            return None
        if self._index_of(frame) != idx:
            self._gen = None
            return None
        self._next = idx + 1
        return frame

    def close(self) -> None:
        """Close the container (idempotent), waiting for any read in flight to finish."""
        with self._lock:
            self._gen = None
            if self._container is not None:
                try:
                    self._container.close()
                except Exception as exc:  # noqa: BLE001
                    log.debug("closing cursor over %s failed: %s", self.path, exc)
                self._container = None


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
        # Multithreaded encode (frame/slice), several times faster than the
        # single-thread default on a multi-core host; the file is unaffected.
        try:
            stream.thread_type = "AUTO"
        except Exception:  # noqa: BLE001 -- some codecs reject it; keep single-thread
            pass
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
        assert self._stream is not None and self._size is not None
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
