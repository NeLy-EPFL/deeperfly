"""The :class:`FrameReader` base class, frame-array helpers, and source dispatch.

A :class:`FrameReader` is the common interface over the two ways deeperfly reads
footage -- a video file (:class:`~deeperfly.io.video.VideoReader`, PyAV) or an
image sequence (:class:`~deeperfly.io.images.ImageSequenceReader`, OpenCV).
:func:`~deeperfly.io.open_reader` resolves a source to the right subclass **once**;
callers then index (``reader[:]``, ``reader[i]``, ``reader[[0,3,5]]``) or stream
(``stream_frames`` / ``stream_blocks``) against that object. A caller that instead
reads single frames in an unpredictable order -- the viewer -- takes a
:class:`FrameCursor` from :meth:`FrameReader.cursor`, which keeps the decoder open
between reads.

:func:`to_numpy` / :func:`to_torch` adapt decoded frames for callers that want a
NumPy array or a torch tensor. :data:`VIDEO_EXTS` / :data:`IMAGE_EXTS` and
:func:`is_video_file` drive the video-vs-image-sequence dispatch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from jaxtyping import Float

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def is_video_file(path: str | Path) -> bool:
    """Whether ``path`` is an existing video file (decoded as a video, not an image
    directory/glob/sequence)."""
    p = Path(path)
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS


def to_numpy(frames) -> np.ndarray:
    """Collapse decoded frames (NumPy / torch tensor) to a NumPy array.

    Parameters
    ----------
    frames
        A NumPy array or a torch tensor (or any array-like).

    Returns
    -------
    np.ndarray
        The frames as a host NumPy array.
    """
    if isinstance(frames, np.ndarray):
        return frames
    if hasattr(frames, "detach"):  # torch.Tensor
        return frames.detach().cpu().numpy()
    return np.asarray(frames)


def to_torch(frames):
    """Hand frames to torch, zero-copy where possible.

    A ``torch.Tensor`` passes through untouched, any other DLPack-capable array is
    wrapped via the DLPack protocol, and NumPy input (what the PyAV reader returns)
    is wrapped on the host via zero-copy ``torch.from_numpy``.

    Parameters
    ----------
    frames
        A torch tensor, a DLPack-capable array, or a NumPy array.

    Returns
    -------
    torch.Tensor
        The frames as a torch tensor (zero-copy where possible).
    """
    import torch

    if isinstance(frames, torch.Tensor):
        return frames
    if hasattr(frames, "__dlpack__"):  # DLPack-capable array
        return torch.from_dlpack(frames)
    return torch.from_numpy(to_numpy(frames))


class FrameReader(ABC):
    """Reads ``(T, H, W, 3)`` uint8 RGB frames from one footage source.

    The two concrete readers -- :class:`~deeperfly.io.video.VideoReader` (PyAV) and
    :class:`~deeperfly.io.images.ImageSequenceReader` (OpenCV) -- resolve their
    source kind once, at construction, rather than on every read.
    :func:`~deeperfly.io.open_reader` is the factory that picks the subclass.

    All decoding runs on the CPU and yields host ``(T, H, W, 3)`` uint8 RGB NumPy.

    Index with ``reader[key]`` to decode frames into an array:

    - ``reader[5]`` -- single frame, ``(H, W, 3)``
    - ``reader[[0, 3, 5]]`` -- explicit indices (random-access), ``(T, H, W, 3)``
    - ``reader[2:8:2]`` -- sequential slice, ``(T, H, W, 3)``
    - ``reader[:]`` -- full decode, ``(T, H, W, 3)``

    Use :meth:`stream_frames` / :meth:`stream_blocks` for lazy forward iteration.

    Readers can be used as context managers (symmetric with
    :class:`~deeperfly.io.video.VideoWriter`); :meth:`close` releases any held
    resources and is a no-op for the stateless readers, which open and close the
    underlying file per operation.
    """

    def __enter__(self) -> FrameReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Release any resources held by the reader (no-op by default)."""

    def cursor(
        self, *, gray_ok: bool = False, same_as_rgb: bool = False
    ) -> FrameCursor:
        """A stateful random-access cursor over this source.

        The base implementation delegates every read to ``self[idx]``, which is correct
        for any source; :class:`~deeperfly.io.video.VideoCursor` overrides it to keep the
        decoder alive between reads.

        Parameters
        ----------
        gray_ok
            Permission, not a promise: the caller accepts single-channel ``(H, W)``
            frames for pictures that carry no color, and a cursor that can tell cheaply
            may return them. A cursor with no cheap way to know returns RGB as always.
        same_as_rgb
            Narrows ``gray_ok`` to frames whose gray is *numerically* what the RGB decode
            would produce. A viewer wants a picture and can take a range-scaled gray; a
            caller doing arithmetic on the pixels -- a detector, a crop search -- must
            keep seeing the numbers it always saw, and passes this.

        Returns
        -------
        FrameCursor
            A cursor the caller owns and should :meth:`~FrameCursor.close`.
        """
        return FrameCursor(self)

    @abstractmethod
    def __getitem__(self, key: int | list[int] | slice) -> Float[np.ndarray, "..."]:
        """Decode frames into a NumPy array.

        Parameters
        ----------
        key
            - ``int`` -- single frame index; returns ``(H, W, 3)`` uint8 RGB.
            - ``list[int]`` -- explicit frame indices (random-access / seeking);
              returns ``(T, H, W, 3)`` in the requested order.
            - ``slice`` -- sequential range ``slice(start, stop, step)``;
              returns ``(T, H, W, 3)``. ``reader[:]`` decodes everything.
        """

    @abstractmethod
    def stream_frames(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
        gray_ok: bool = False,
        thread_count: int | None = None,
    ) -> Iterator[Float[np.ndarray, "H W 3"]]:
        """Yield individual ``(H, W, 3)`` uint8 RGB frames from one forward pass.

        Parameters
        ----------
        start, stop, step
            Frame range, like ``range(start, stop, step)``.
        gray_ok, thread_count
            See :meth:`stream_blocks`.
        """

    @abstractmethod
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
        """Yield ``(T, H, W, 3)`` uint8 RGB blocks from one forward pass.

        Instead of decoding a fixed ``[start, stop)`` slice, walk the source forward
        and emit frames in groups of up to ``block_size``. A whole recording is
        therefore one linear decode -- no per-window re-open or re-seek.

        Parameters
        ----------
        start, stop, step
            Frame range, like ``range(start, stop, step)``.
        block_size
            Maximum frames per yielded block.
        gray_ok
            Permission, not a promise: the caller accepts ``(T, H, W, 1)`` blocks for
            footage that carries no color. The channel axis is **kept** (unlike
            :meth:`cursor`, which drops it) so that ops and models indexing it from the
            right work unchanged. A source that cannot tell cheaply returns RGB as always,
            so a caller must read ``shape[-1]`` rather than assume which it got.
        thread_count
            Decode threads for this stream, or ``None`` for the backend's own choice.
            Worth setting when several sources are decoded at once: sizing each pool from
            the core count oversubscribes the host.

        Yields
        ------
        np.ndarray
            ``(T, H, W, C)`` uint8 blocks with ``T <= block_size`` and ``C`` 3, or 1 when
            ``gray_ok`` was granted and the picture has no color.
        """

    @abstractmethod
    def count(self) -> int | None:
        """Best-effort frame count -- ``None`` when unknown.

        A **hint** for a progress-bar total only: callers stream frames and detect
        end-of-file from the decoder itself, so an off-by-a-few count or ``None``
        never affects correctness.
        """

    def fps(self) -> float | None:
        """Frame rate in frames/sec, or ``None`` when unknown.

        Image sequences carry no intrinsic frame rate, so the base implementation
        returns ``None``; :class:`~deeperfly.io.video.VideoReader` overrides it.
        """
        return None


class FrameCursor:
    """One frame at a time from one source, holding what makes the next read cheap.

    Indexing a reader is *stateless*: it opens the source, decodes, and lets go again.
    That is the right trade for a batch pass, which reads each frame once and walks
    forward. A viewer is the opposite -- it reads one frame per camera, in an order the
    operator chooses, and pays that setup on every single frame. A cursor is the stateful
    counterpart: open it once, ask it for frames in any order, close it when the source is
    no longer being shown.

    This base implementation just delegates to the reader, so *every* source has a cursor
    and a caller never has to know which kind it is holding.
    :class:`~deeperfly.io.video.VideoCursor` is the one that actually holds a decoder open.

    Cursors are single-source and may be shared across threads only if the subclass says
    so (:class:`~deeperfly.io.video.VideoCursor` locks; this one inherits whatever
    ``reader[idx]`` guarantees).
    """

    def __init__(self, reader: FrameReader) -> None:
        self._reader = reader

    def frame(self, idx: int) -> np.ndarray:
        """Frame ``idx`` as ``(H, W, 3)`` uint8 RGB, or ``(H, W)`` if it has no color.

        A 2-D result is only ever returned when the cursor was opened with ``gray_ok``
        (see :meth:`FrameReader.cursor`), so callers that did not ask for it can treat
        the result as RGB.
        """
        return self._reader[idx]

    def close(self) -> None:
        """Release anything the cursor holds open (no-op by default)."""
