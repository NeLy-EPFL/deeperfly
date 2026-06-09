"""Image-sequence reading: :class:`ImageSequenceReader` plus the still-image registry.

Image *sequences* (not video files) are decoded per file by OpenCV, resolved by
:func:`select_image_reader`. Decoding is parallel across threads (JPEG/PNG decoders
release the GIL) and yields host ``(T, H, W, 3)`` uint8 RGB NumPy: grayscale frames
broadcast to 3 channels, alpha is dropped.
"""

from __future__ import annotations

import glob
import importlib.util
import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from jaxtyping import Float

from .base import IMAGE_EXTS, FrameReader

log = logging.getLogger("deeperfly.io")


def _have(*modules: str) -> bool:
    """True if every module can be located without importing the heavy parts."""
    for mod in modules:
        try:
            if importlib.util.find_spec(mod) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


IMAGE_READ_ORDER = ("opencv",)
_IMAGE_READER_REQUIRES = {"opencv": ("cv2",)}


def list_image_readers() -> list[str]:
    """All known image-reader names (installed or not)."""
    return sorted(IMAGE_READ_ORDER)


def available_image_readers() -> list[str]:
    """Image-reader names whose dependencies are importable in this environment."""
    return sorted(n for n in IMAGE_READ_ORDER if _have(*_IMAGE_READER_REQUIRES[n]))


def select_image_reader(backend: str = "auto") -> str:
    """Resolve an image-reader name (or ``"auto"``).

    Parameters
    ----------
    backend
        ``"auto"`` or ``"opencv"``.

    Returns
    -------
    str
        The resolved image-reader name (always ``"opencv"``).

    Raises
    ------
    ValueError
        If ``backend`` names no known image reader.
    RuntimeError
        If OpenCV is unavailable.
    """
    if backend == "auto":
        if _have(*_IMAGE_READER_REQUIRES["opencv"]):
            return "opencv"
        raise RuntimeError("no image reader available; install opencv-python")
    if backend not in _IMAGE_READER_REQUIRES:
        raise ValueError(
            f"unknown image reader {backend!r}; choose from {list_image_readers()}"
        )
    if not _have(*_IMAGE_READER_REQUIRES[backend]):
        raise RuntimeError(
            f"image reader {backend!r} needs {_IMAGE_READER_REQUIRES[backend]}"
        )
    return backend


def list_image_files(pattern: str | Path) -> list[Path]:
    """Sorted image files for a directory or glob pattern (by name).

    Parameters
    ----------
    pattern
        A directory of images, or a glob pattern.

    Returns
    -------
    list of Path
        The matching image files, sorted by name.

    Raises
    ------
    FileNotFoundError
        If nothing matches ``pattern``.
    """
    p = Path(pattern)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    else:
        files = sorted(Path(f) for f in glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"no images matched {pattern!r}")
    return files


class ImageSequenceReader(FrameReader):
    """Reads an ordered image sequence (a directory, glob, or explicit file list).

    Decode-thread count is fixed at construction. Frames are decoded in parallel
    across threads via OpenCV; the result is host ``(T, H, W, 3)`` uint8 RGB.
    Image sequences carry no frame rate, so :meth:`fps` is the inherited ``None``.
    """

    def __init__(
        self,
        files,
        *,
        workers: int | None = None,
    ) -> None:
        self.files = [Path(f) for f in files]
        self.workers = workers

    @classmethod
    def from_pattern(
        cls,
        pattern: str | Path,
        *,
        workers: int | None = None,
    ) -> ImageSequenceReader:
        """Build a reader for a directory or glob, listing/sorting its files by name."""
        return cls(list_image_files(pattern), workers=workers)

    @property
    def name(self) -> str:
        return "opencv"

    # -- decode (parallel per-file, CPU) -------------------------------------

    @staticmethod
    def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
        """Coerce a decoded image to ``(H, W, 3)`` uint8 (grayscale broadcast, alpha dropped)."""
        if arr.ndim == 2:  # grayscale (H, W) -> (H, W, 1)
            arr = arr[..., None]
        if arr.shape[-1] == 1:  # single channel -> 3 (broadcast, not a width slice!)
            arr = np.repeat(arr, 3, axis=-1)
        arr = arr[..., :3]  # drop alpha / extra channels
        return arr if arr.dtype == np.uint8 else np.clip(arr, 0, 255).astype(np.uint8)

    @staticmethod
    def _select(files: list[Path], indices, start, stop, step) -> list[Path]:
        """Pick frames by explicit ``indices`` or a ``range(start, stop, step)`` slice."""
        if indices is not None:
            return [files[int(i)] for i in indices]
        return files[start : stop if stop is not None else len(files) : step]

    def _n_workers(self, n: int) -> int:
        return max(1, min(n, self.workers or (os.cpu_count() or 4)))

    def _decode(self, files: list[Path]) -> np.ndarray:
        """Parallel CPU decode of ``files`` -> ``(T, H, W, 3)`` uint8 NumPy."""
        import cv2

        def decode(f: Path) -> np.ndarray:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR_RGB)  # (H, W, 3) RGB uint8
            if img is not None:
                return self._to_rgb_uint8(img)
            raise OSError(f"failed to decode image: {f} (OpenCV returned None)")

        with ThreadPoolExecutor(max_workers=self._n_workers(len(files))) as pool:
            frames = list(pool.map(decode, files))
        return np.stack(frames)

    def __getitem__(self, key: int | list[int] | slice) -> Float[np.ndarray, "..."]:
        if isinstance(key, int):
            files = [self.files[key]]
            out = self._decode(files)[0]
        elif isinstance(key, list):
            if not key:
                raise ValueError("index list must be non-empty")
            files = self._select(self.files, key, 0, None, 1)
            out = self._decode(files)
        elif isinstance(key, slice):
            start, stop, step = key.start or 0, key.stop, key.step or 1
            files = self._select(self.files, None, start, stop, step)
            if not files:
                raise ValueError("no frames selected (check slice)")
            out = self._decode(files)
        else:
            raise TypeError(f"invalid index type {type(key).__name__!r}")
        log.debug(
            "read images (%s) -> %s",
            self.name,
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
        files = self._select(self.files, None, start, stop, step)
        for f in files:
            yield self._decode([f])[0]

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
        files = self._select(self.files, None, start, stop, step)
        for pos in range(0, len(files), block_size):
            yield self._decode(files[pos : pos + block_size])

    def count(self) -> int | None:
        return len(self.files)  # image sequence: one frame per file
