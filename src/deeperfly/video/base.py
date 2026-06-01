"""Pluggable video backend registry: readers (CPU/GPU) and writers.

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

Frame contract: ``(T, H, W, 3)`` ``uint8`` RGB. CPU backends return NumPy;
GPU backends (``torchcodec``, ``decord``, ``pynvvideocodec``, ``dali``) keep
frames as a framework tensor on the requested device. :func:`to_numpy` collapses
either to NumPy on the host; :func:`to_jax` hands GPU frames to JAX zero-copy.
"""

from __future__ import annotations

import abc
import importlib.util
from collections.abc import Sequence

import numpy as np

# Registries (name -> backend class), populated by the decorators below.
_READERS: dict[str, type["ReaderBackend"]] = {}
_WRITERS: dict[str, type["WriterBackend"]] = {}

# Preference order for ``backend="auto"``. Explicit names always win; "auto"
# walks these and picks the first installed backend. CPU order keeps imageio
# first so existing behavior is unchanged when nothing else is installed.
CPU_READ_ORDER = ("imageio", "pyav", "opencv", "decord", "video_reader_rs")
GPU_READ_ORDER = ("torchcodec", "decord", "pynvvideocodec", "dali")
WRITE_ORDER = ("imageio", "pyav", "opencv")


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


def is_gpu_device(device) -> bool:
    """Whether ``device`` names a non-CPU device (``"cuda"``, ``"cuda:0"``...)."""
    return device is not None and str(device).split(":")[0] not in ("cpu", "")


def device_id(device) -> int:
    """GPU ordinal from a device string (``"cuda:2"`` -> 2, ``"cuda"`` -> 0)."""
    parts = str(device).split(":")
    return int(parts[1]) if len(parts) > 1 and parts[1] else 0


def require_cpu(device, name: str) -> None:
    if is_gpu_device(device):
        raise ValueError(f"{name!r} is a CPU-only backend; got device={device!r}")


class ReaderBackend(abc.ABC):
    """Decode an encoded video into ``(T, H, W, 3)`` frames."""

    name: str = ""
    requires: tuple[str, ...] = ()
    supports_gpu: bool = False
    supports_seek: bool = False  # True if random access seeks rather than scans

    @classmethod
    def is_available(cls) -> bool:
        return _have(*cls.requires)

    @classmethod
    def read(cls, path, *, device="cpu", start=0, stop=None, step=1, indices=None):
        """Sequential range, or random access when ``indices`` is given."""
        if indices is not None:
            idx = [int(i) for i in indices]
            if not idx:
                raise ValueError("indices must be a non-empty sequence")
            return cls._read_indices(path, device, idx)
        return cls._read_sequential(path, device, int(start), stop, int(step))

    @staticmethod
    @abc.abstractmethod
    def _read_sequential(path, device, start, stop, step):
        """Decode ``range(start, stop, step)`` (NumPy on CPU, tensor on GPU)."""

    @classmethod
    def _read_indices(cls, path, device, indices: Sequence[int]):
        """Gather arbitrary frame ``indices``. Default: decode-once-and-select."""
        frames = cls._read_sequential(path, device, 0, max(indices) + 1, 1)
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


def select_reader(backend: str = "auto", *, device="cpu") -> type[ReaderBackend]:
    """Resolve a reader backend by name (or ``"auto"``) for the given device."""
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
        if is_gpu_device(device) and not cls.supports_gpu:
            require_cpu(device, backend)
        return cls

    order = GPU_READ_ORDER if is_gpu_device(device) else CPU_READ_ORDER
    for name in order:
        cls = _READERS.get(name)
        if cls is not None and cls.is_available():
            return cls
    kind = "GPU" if is_gpu_device(device) else "CPU"
    raise RuntimeError(
        f"no {kind} video read backend available; install one of {list(order)}"
    )


def select_writer(backend: str = "auto") -> type[WriterBackend]:
    """Resolve a writer backend by name (or ``"auto"``)."""
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
    """Collapse backend frames (NumPy / torch / DALI tensor) to a NumPy array."""
    if isinstance(frames, np.ndarray):
        return frames
    if hasattr(frames, "detach"):  # torch.Tensor
        return frames.detach().cpu().numpy()
    if hasattr(frames, "as_cpu"):  # DALI TensorList(GPU)
        frames = frames.as_cpu()
    if hasattr(frames, "as_array"):  # DALI TensorList
        return np.asarray(frames.as_array())
    if hasattr(frames, "asnumpy"):  # decord NDArray
        return frames.asnumpy()
    return np.asarray(frames)


def to_jax(frames):
    """Hand backend frames to JAX, zero-copy where possible.

    GPU tensors (e.g. from the ``torchcodec`` or ``decord`` readers) are wrapped
    via the DLPack protocol, so on a shared GPU JAX reads the *same* device
    buffer with no host round-trip -- the fast path for feeding a JAX detector::

        frames = video.read_video("clip.mp4", backend="torchcodec", device="cuda")
        x = video.to_jax(frames)          # jax.Array on the GPU, zero-copy

    NumPy input is moved onto JAX's default device the usual way. Keep a
    reference to the producer tensor until JAX has consumed it.
    """
    import jax.numpy as jnp

    if isinstance(frames, np.ndarray):
        return jnp.asarray(frames)
    if hasattr(frames, "__dlpack__"):  # torch / decord / most array libs
        return jnp.from_dlpack(frames)
    if hasattr(frames, "to_dlpack"):  # decord NDArray (older API)
        return jnp.from_dlpack(frames.to_dlpack())
    return jnp.asarray(to_numpy(frames))


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
