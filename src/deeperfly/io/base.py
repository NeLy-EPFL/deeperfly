"""The :class:`FrameReader` base class, frame-array helpers, and source dispatch.

A :class:`FrameReader` is the common interface over the two ways deeperfly reads
footage -- a video file (:class:`~deeperfly.io.video.VideoReader`, PyAV) or an
image sequence (:class:`~deeperfly.io.images.ImageSequenceReader`, OpenCV/imageio).
:func:`~deeperfly.io.open_reader` resolves a source to the right subclass **once**;
callers then ``read`` / ``stream`` / ``count`` / ``fps`` against that object.

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
    :class:`~deeperfly.io.images.ImageSequenceReader` (OpenCV/imageio) -- resolve
    their source kind at construction, so the per-call dispatch the old free
    functions repeated happens once. :func:`~deeperfly.io.open_reader` is the
    factory that picks the subclass.

    All decoding runs on the CPU and yields host ``(T, H, W, 3)`` uint8 RGB NumPy.

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

    @property
    @abstractmethod
    def name(self) -> str:
        """The decoder this reader actually uses (e.g. ``"pyav"``/``"opencv"``), for
        logs and diagnostics."""

    @abstractmethod
    def read(
        self,
        *,
        indices: list[int] | None = None,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Float[np.ndarray, "T H W 3"]:
        """Decode a frame selection into a stacked ``(T, H, W, 3)`` uint8 RGB array.

        Parameters
        ----------
        indices
            Explicit frame indices (random access); overrides ``start/stop/step``.
        start, stop, step
            Sequential frame slice, like ``range(start, stop, step)``.

        Returns
        -------
        np.ndarray
            The decoded ``(T, H, W, 3)`` uint8 RGB frames (host NumPy).
        """

    @abstractmethod
    def stream(self, *, block: int = 64) -> Iterator[Float[np.ndarray, "T H W 3"]]:
        """Yield ``(T, H, W, 3)`` uint8 RGB blocks (``T <= block``) from one forward pass.

        Instead of decoding a fixed ``[start, stop)`` slice, walk the source forward
        to the end, emitting frames in groups of up to ``block``. A whole recording
        is therefore one linear decode -- no per-window re-open or re-seek -- which
        is what streaming detection wants. ``block`` only sets the grouping
        granularity; it does not bound how far decode runs ahead (the consumer
        imposes that backpressure).

        Parameters
        ----------
        block
            Maximum frames per yielded block.

        Yields
        ------
        np.ndarray
            ``(T, H, W, 3)`` uint8 RGB blocks with ``T <= block``.
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
