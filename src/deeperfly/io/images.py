"""Image-sequence reading: :class:`ImageSequenceReader` (OpenCV).

Image *sequences* (not video files) are decoded per file by OpenCV. Decoding is
parallel across threads (JPEG/PNG decoders release the GIL) and yields host
``(T, H, W, 3)`` uint8 RGB NumPy: grayscale frames broadcast to 3 channels,
alpha is dropped.
"""

from __future__ import annotations

import glob
import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from jaxtyping import Float

from .base import IMAGE_EXTS, FrameReader

log = logging.getLogger("deeperfly.io")


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
            out = self._decode([self.files[key]])[0]
        elif isinstance(key, list):
            if not key:
                raise ValueError("index list must be non-empty")
            out = self._decode([self.files[int(i)] for i in key])
        elif isinstance(key, slice):
            files = self.files[key]
            if not files:
                raise ValueError("no frames selected (check slice)")
            out = self._decode(files)
        else:
            raise TypeError(f"invalid index type {type(key).__name__!r}")
        log.debug("read images -> %s", out.shape)
        return out

    def stream_frames(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Iterator[Float[np.ndarray, "H W 3"]]:
        for f in self.files[start:stop:step]:
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
        files = self.files[start:stop:step]
        for pos in range(0, len(files), block_size):
            yield self._decode(files[pos : pos + block_size])

    def count(self) -> int | None:
        return len(self.files)  # image sequence: one frame per file
