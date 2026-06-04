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
GPU backends (``torchcodec``, ``decord``, ``dali``) keep
frames as a framework tensor on the requested device. :func:`to_numpy` collapses
either to NumPy on the host; :func:`to_torch` hands GPU frames to torch zero-copy.
"""

from __future__ import annotations

import abc
import importlib.util
from collections.abc import Sequence

import numpy as np

# Registries (name -> backend class), populated by the decorators below.
_READERS: dict[str, type["ReaderBackend"]] = {}
_WRITERS: dict[str, type["WriterBackend"]] = {}

# Preference order for ``backend="auto"`` -- fastest first. Explicit names always
# win; "auto" walks the list for the device and picks the first *installed*
# backend. CPU order leads with pyav, the in-process core default (it links FFmpeg
# directly -- no subprocess), then the other in-process decoders (opencv /
# video_reader_rs / torchcodec / decord), and keeps imageio last: it shells out to
# the ``ffmpeg`` binary (a subprocess fork -- slower, and it trips Python 3.13's
# os.fork()-in-a-multithreaded-process warning once JAX has started threads), so
# it is only the optional forking fallback. GPU order leads with torchcodec -- the
# most robust NVDEC path and the one that feeds torch zero-copy (see ``to_torch``);
# every listed GPU decoder (torchcodec / DALI / decord) is frame-accurate.
CPU_READ_ORDER = (
    "pyav",
    "opencv",
    "torchcodec",
    "decord",
    "video_reader_rs",
    "imageio",
)
# torchcodec (fastest, when its CUDA build + NPP are present) -> DALI (robust
# NVDEC, needs only the driver) -> decord (GPU only with a rare CUDA build).
GPU_READ_ORDER = ("torchcodec", "dali", "decord")
# Writers prefer pyav, the core default: like imageio it encodes H.264 (libx264),
# but in-process -- imageio shells out to the ``ffmpeg`` binary (the os.fork()
# subprocess warning again, now at render time). Fall back to imageio (libx264,
# but forks) and then opencv (mp4v fourcc, last resort) when pyav is absent.
WRITE_ORDER = ("pyav", "imageio", "opencv")


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


# Friendly device aliases: users (and our example config) naturally write "gpu",
# but torch/torchcodec only understand "cuda" -- ``device="gpu"`` reaches the
# decoder verbatim and dies with "Unknown device type: gpu". Normalize at the read
# boundary so "gpu" / "gpu:1" mean "cuda" / "cuda:1"; every other value (cpu, cuda,
# auto, ...) passes through untouched.
_DEVICE_ALIASES = {"gpu": "cuda"}


def canonical_device(device):
    """Map the ``"gpu"`` alias to ``"cuda"`` (preserving any ``:ordinal``); else unchanged."""
    if device is None:
        return device
    head, sep, tail = str(device).partition(":")
    alias = _DEVICE_ALIASES.get(head)
    return f"{alias}{sep}{tail}" if alias else device


def is_gpu_device(device) -> bool:
    """Whether ``device`` names a non-CPU device (``"cuda"``, ``"cuda:0"``, ``"gpu"``...)."""
    return device is not None and str(device).split(":")[0] not in ("cpu", "")


def device_id(device) -> int:
    """GPU ordinal from a device string (``"cuda:2"`` -> 2, ``"cuda"`` -> 0)."""
    parts = str(device).split(":")
    return int(parts[1]) if len(parts) > 1 and parts[1] else 0


def require_cpu(device, name: str) -> None:
    if is_gpu_device(device):
        raise ValueError(f"{name!r} is a CPU-only backend; got device={device!r}")


def cuda_available() -> bool:
    """Whether a CUDA GPU is usable, for resolving ``device="auto"``.

    Probes torch (a core dependency, so always importable). The probe does not
    initialize JAX; ``torch.cuda`` is cheap once torch is loaded and any import
    cost is paid once per process.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# Process-wide cache of how GPU "auto" decode actually behaves here, so we probe
# the environment once instead of on every read. ``_gpu_auto_failed`` is set when
# *every* installed GPU backend fails to decode (a GPU is present but none can use
# it -- e.g. a torchcodec build without working CUDA); after that, "auto" steers
# clear of the GPU. ``_gpu_auto_reader`` records the first GPU backend that *did*
# decode, so later reads go straight to it without re-trying the broken ones.
_gpu_auto_failed = False
_gpu_auto_reader: str | None = None


def mark_gpu_auto_failed() -> None:
    """Disable GPU selection for ``device="auto"`` for the rest of the process."""
    global _gpu_auto_failed
    _gpu_auto_failed = True


def remember_gpu_reader(name: str) -> None:
    """Record the GPU backend now known to decode here (skip the rest next time)."""
    global _gpu_auto_reader
    _gpu_auto_reader = name


def gpu_reader_candidates(backend: str = "auto") -> list[type["ReaderBackend"]]:
    """GPU reader classes to try, in order, for ``device`` on the GPU.

    A forced ``backend`` yields just that one (strict). ``"auto"`` yields the
    cached winner if one is known, else every installed backend from
    :data:`GPU_READ_ORDER` -- so :func:`read_video` can walk them until one
    actually decodes.
    """
    _ensure_backends()
    if backend != "auto":
        cls = _READERS.get(backend)
        return [cls] if cls is not None else []
    if _gpu_auto_reader is not None:
        return [_READERS[_gpu_auto_reader]]
    return [
        cls
        for name in GPU_READ_ORDER
        if (cls := _READERS.get(name)) is not None and cls.is_available()
    ]


def resolve_device(device, backend: str = "auto") -> str:
    """Resolve ``device="auto"`` to a concrete ``"cuda"`` or ``"cpu"``.

    ``"auto"`` picks ``"cuda"`` when a GPU is present *and* a GPU-capable read
    backend is usable -- either an installed backend from :data:`GPU_READ_ORDER`
    (when ``backend="auto"``), or the forced ``backend`` if it advertises
    ``supports_gpu``. Otherwise it falls back to ``"cpu"``. Concrete device
    strings (``"cpu"``, ``"cuda:1"``, ...) pass through unchanged -- except an
    ``auto``-backend GPU request is downgraded to ``"cpu"`` once a prior probe has
    shown no GPU backend can decode here (so callers stop retrying a dead path).
    """
    device = canonical_device(device)  # "gpu" -> "cuda" before anyone sees it
    if device != "auto":
        if backend == "auto" and _gpu_auto_failed and is_gpu_device(device):
            return "cpu"
        return device
    if _gpu_auto_failed or not cuda_available():
        return "cpu"
    _ensure_backends()
    if backend == "auto":
        gpu_ready = any(
            (cls := _READERS.get(name)) is not None and cls.is_available()
            for name in GPU_READ_ORDER
        )
    else:
        cls = _READERS.get(backend)
        gpu_ready = cls is not None and cls.supports_gpu
    return "cuda" if gpu_ready else "cpu"


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
    """Resolve a reader backend by name (or ``"auto"``) for the given device.

    ``device="auto"`` is resolved first (:func:`resolve_device`): it becomes a
    GPU device when hardware and a GPU backend are both available, else CPU.
    """
    _ensure_backends()
    device = resolve_device(device, backend)
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


def to_torch(frames):
    """Hand backend frames to torch, zero-copy where possible.

    GPU tensors (e.g. from the ``torchcodec`` or ``decord`` readers) stay on the
    device: a ``torch.Tensor`` passes through and any other DLPack-capable tensor
    is wrapped via the DLPack protocol, so torch reads the *same* device buffer
    with no host round-trip -- the fast path for feeding the detector::

        frames = video.read_video("clip.mp4", backend="torchcodec", device="cuda")
        x = video.to_torch(frames)        # torch.Tensor on the GPU, zero-copy

    NumPy input is wrapped on the host the usual way. Keep a reference to the
    producer tensor until torch has consumed it.
    """
    import torch

    if isinstance(frames, torch.Tensor):
        return frames
    if hasattr(frames, "__dlpack__"):  # decord / most array libs
        return torch.from_dlpack(frames)
    if hasattr(frames, "to_dlpack"):  # decord NDArray (older API)
        return torch.from_dlpack(frames.to_dlpack())
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
