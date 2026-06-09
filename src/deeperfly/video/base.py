"""Pluggable video backend registry: readers and writers (all CPU).

A *backend* is a small class declaring the modules it needs and how to decode
or encode frames. Backends register themselves on import (via
:func:`register_reader` / :func:`register_writer`) and are resolved lazily, so a
missing optional dependency only matters when that backend is actually chosen.

Readers expose two access patterns:

- **sequential** -- ``read(..., start, stop, step)`` decodes a frame range.
- **random access** -- ``read(..., indices=[...])`` returns those frames in the
  requested order. Backends advertise ``supports_seek``: when ``True`` they seek
  to each frame (cheap for sparse picks); otherwise the base falls back to
  decoding up to ``max(indices)`` once and gathering, which is always correct.

Frame contract: ``(T, H, W, 3)`` ``uint8`` RGB. All decoding happens on the CPU:
backends return NumPy, except ``torchcodec``, which returns a CPU torch tensor.
:func:`to_numpy` collapses either to a NumPy array; :func:`to_torch` hands frames
to torch.
"""

from __future__ import annotations

import abc
import importlib.util
from collections.abc import Sequence

import numpy as np

# Registries (name -> backend class), populated by the decorators below.
_READERS: dict[str, type["ReaderBackend"]] = {}
_WRITERS: dict[str, type["WriterBackend"]] = {}

# Preference order for ``backend="auto"`` -- fastest first; "auto" picks the first
# *installed* one. Leads with pyav, the in-process core default (links FFmpeg
# directly), then opencv / torchcodec / video_reader_rs. All decode on the CPU.
READ_ORDER = (
    "pyav",
    "opencv",
    "torchcodec",
    "video_reader_rs",
)
# Writers prefer pyav (in-process H.264 via libx264), falling back to opencv (mp4v
# fourcc) when pyav is absent.
WRITE_ORDER = ("pyav", "opencv")


def register_reader(cls: type["ReaderBackend"]) -> type["ReaderBackend"]:
    _READERS[cls.name] = cls
    return cls


def register_writer(cls: type["WriterBackend"]) -> type["WriterBackend"]:
    _WRITERS[cls.name] = cls
    return cls


def _have(*modules: str) -> bool:
    """True if every module can be located without importing the heavy parts."""
    for mod in modules:
        try:
            if importlib.util.find_spec(mod) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


#: Frames per block when the default :meth:`ReaderBackend.stream` walks a source
#: in chunks (seek-based backends seek to each block, so this stays linear).
STREAM_BLOCK = 256


class ReaderBackend(abc.ABC):
    """Decode an encoded video into ``(T, H, W, 3)`` frames (on the CPU)."""

    name: str = ""
    requires: tuple[str, ...] = ()
    supports_seek: bool = False  # True if random access seeks rather than scans

    @classmethod
    def is_available(cls) -> bool:
        return _have(*cls.requires)

    @classmethod
    def read(cls, path, *, start=0, stop=None, step=1, indices=None):
        """Sequential range, or random access when ``indices`` is given.

        Parameters
        ----------
        path
            The video file to decode.
        start, stop, step
            Sequential frame slice (used when ``indices`` is ``None``).
        indices
            Explicit frame indices for random access (overrides the slice).

        Returns
        -------
        np.ndarray
            The decoded ``(T, H, W, 3)`` uint8 frames.
        """
        if indices is not None:
            idx = [int(i) for i in indices]
            if not idx:
                raise ValueError("indices must be a non-empty sequence")
            return cls._read_indices(path, idx)
        return cls._read_sequential(path, int(start), stop, int(step))

    @staticmethod
    @abc.abstractmethod
    def _read_sequential(path, start, stop, step):
        """Decode ``range(start, stop, step)`` to ``(T, H, W, 3)`` uint8."""

    @classmethod
    def stream(cls, path, *, start=0, step=1):
        """Yield single ``(H, W, 3)`` frames from one forward pass to end-of-stream.

        The streaming counterpart to :meth:`read`: a *pull-based, forward-only*
        decode that opens the source once and walks it to the end, instead of
        materializing a bounded ``[start, stop)`` slice. A consumer that wants the
        whole recording gets it in a single linear decode (no per-window re-open),
        and decides for itself how far to pull -- the natural fit for streaming
        detection, and the same contract a future *live-camera* source would
        satisfy (frames arrive forward in time, the total is unknown, there is no
        seek). Hence no ``stop`` and no frame count here.

        The default drives :meth:`_read_sequential` in blocks of
        :data:`STREAM_BLOCK`: correct and linear for seek-based backends (they seek
        to each block's start). Backends whose sequential decode rescans from frame
        0 (pyav, opencv) override this with a true single open-and-walk generator,
        so a full read stays linear rather than quadratic.

        Parameters
        ----------
        path
            The video file to decode.
        start
            First frame to emit.
        step
            Stride between emitted frames.

        Yields
        ------
        np.ndarray
            Single ``(H, W, 3)`` uint8 RGB frames, in order.
        """
        pos = start
        while True:
            try:
                block = cls._read_sequential(path, pos, pos + STREAM_BLOCK * step, step)
            except ValueError:  # decoded no frames -> past end-of-stream
                return
            yield from block
            if len(block) < STREAM_BLOCK:  # a short block is the last one
                return
            pos += STREAM_BLOCK * step

    @classmethod
    def _read_indices(cls, path, indices: Sequence[int]):
        """Gather arbitrary frame ``indices``. Default: decode-once-and-select."""
        frames = cls._read_sequential(path, 0, max(indices) + 1, 1)
        return frames[list(indices)]


class WriterBackend(abc.ABC):
    """Encode ``(T, H, W, 3)`` ``uint8`` RGB frames to a video file."""

    name: str = ""
    requires: tuple[str, ...] = ()

    @classmethod
    def is_available(cls) -> bool:
        return _have(*cls.requires)

    @staticmethod
    @abc.abstractmethod
    def write(frames, path, *, fps=30.0, codec=None, **kwargs): ...


def _ensure_backends() -> None:
    """Import the backend modules so their registration side effects run."""
    from . import backends  # noqa: F401  (import for side effects)


def select_reader(backend: str = "auto") -> type[ReaderBackend]:
    """Resolve a reader backend by name (or ``"auto"``).

    ``"auto"`` walks :data:`READ_ORDER` and returns the first installed backend.

    Parameters
    ----------
    backend
        A reader name, or ``"auto"``.

    Returns
    -------
    type[ReaderBackend]
        The resolved backend class.

    Raises
    ------
    ValueError
        If ``backend`` names no known reader.
    RuntimeError
        If the named (or every auto-order) backend is unavailable.
    """
    _ensure_backends()
    if backend != "auto":
        try:
            cls = _READERS[backend]
        except KeyError:
            raise ValueError(
                f"unknown read backend {backend!r}; choose from {list_read_backends()}"
            ) from None
        if not cls.is_available():
            raise RuntimeError(
                f"read backend {backend!r} needs {cls.requires}; install it"
            )
        return cls

    for name in READ_ORDER:
        cls = _READERS.get(name)
        if cls is not None and cls.is_available():
            return cls
    raise RuntimeError(
        f"no video read backend available; install one of {list(READ_ORDER)}"
    )


def select_writer(backend: str = "auto") -> type[WriterBackend]:
    """Resolve a writer backend by name (or ``"auto"``).

    Parameters
    ----------
    backend
        A writer name, or ``"auto"``.

    Returns
    -------
    type[WriterBackend]
        The resolved backend class.

    Raises
    ------
    ValueError
        If ``backend`` names no known writer.
    RuntimeError
        If the named (or every auto-order) backend is unavailable.
    """
    _ensure_backends()
    if backend != "auto":
        try:
            cls = _WRITERS[backend]
        except KeyError:
            raise ValueError(
                f"unknown write backend {backend!r}; choose from {list_write_backends()}"
            ) from None
        if not cls.is_available():
            raise RuntimeError(
                f"write backend {backend!r} needs {cls.requires}; install it"
            )
        return cls
    for name in WRITE_ORDER:
        cls = _WRITERS.get(name)
        if cls is not None and cls.is_available():
            return cls
    raise RuntimeError(
        f"no video write backend available; install one of {list(WRITE_ORDER)}"
    )


def to_numpy(frames) -> np.ndarray:
    """Collapse backend frames (NumPy / torch tensor) to a NumPy array.

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
    """Hand backend frames to torch, zero-copy where possible.

    A ``torch.Tensor`` (e.g. from the ``torchcodec`` reader) passes through
    untouched, and any other DLPack-capable array is wrapped via the DLPack
    protocol; NumPy input is wrapped on the host the usual way.

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


def list_read_backends() -> list[str]:
    """All registered reader names (installed or not)."""
    _ensure_backends()
    return sorted(_READERS)


def list_write_backends() -> list[str]:
    _ensure_backends()
    return sorted(_WRITERS)


def available_read_backends() -> list[str]:
    """Reader names whose dependencies are importable in this environment."""
    _ensure_backends()
    return sorted(n for n, c in _READERS.items() if c.is_available())


def available_write_backends() -> list[str]:
    _ensure_backends()
    return sorted(n for n, c in _WRITERS.items() if c.is_available())


# Still-image decode (image sequences, not video files). These aren't full backend
# classes -- the work is a per-file decode -- so the registry is just a preference
# order plus a name resolver. ``opencv`` is the core default; ``imageio`` is an
# optional broad-format fallback (see ``deeperfly.video.io._read_images_cpu``).
IMAGE_READ_ORDER = ("opencv", "imageio")
_IMAGE_READER_REQUIRES = {"opencv": ("cv2",), "imageio": ("imageio",)}


def list_image_readers() -> list[str]:
    """All known image-reader names (installed or not)."""
    return sorted(IMAGE_READ_ORDER)


def available_image_readers() -> list[str]:
    """Image-reader names whose dependencies are importable in this environment."""
    return sorted(n for n in IMAGE_READ_ORDER if _have(*_IMAGE_READER_REQUIRES[n]))


def select_image_reader(backend: str = "auto") -> str:
    """Resolve an image-reader name (or ``"auto"``).

    ``"auto"`` walks :data:`IMAGE_READ_ORDER` and returns the first installed reader
    (``opencv`` is a core dependency, so this is normally ``"opencv"``).

    Parameters
    ----------
    backend
        An image-reader name, or ``"auto"``.

    Returns
    -------
    str
        The resolved image-reader name.

    Raises
    ------
    ValueError
        If ``backend`` names no known image reader.
    RuntimeError
        If the named (or every auto-order) reader is unavailable.
    """
    if backend == "auto":
        for name in IMAGE_READ_ORDER:
            if _have(*_IMAGE_READER_REQUIRES[name]):
                return name
        raise RuntimeError(
            f"no image reader available; install one of {list(IMAGE_READ_ORDER)}"
        )
    if backend not in _IMAGE_READER_REQUIRES:
        raise ValueError(
            f"unknown image reader {backend!r}; choose from {list_image_readers()}"
        )
    if not _have(*_IMAGE_READER_REQUIRES[backend]):
        raise RuntimeError(
            f"image reader {backend!r} needs {_IMAGE_READER_REQUIRES[backend]}; "
            "install it (e.g. the optional 'imageio' extra)"
        )
    return backend
