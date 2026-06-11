"""Locate each camera's footage and decode frames on demand for the viewer.

The footage paths recorded in ``results.h5`` (:meth:`StageStore.read_footage`)
are tried in order -- resolved-absolute, then relative to the ``results.h5``
directory, then a user-supplied directory (by file name) -- so a result moved or
copied still finds its videos. :class:`FrameSource` opens one
:class:`~deeperfly.io.base.FrameReader` per camera and caches recently decoded
frames; a camera whose footage cannot be found yields a black frame sized from
the recorded ``image_sizes`` so the skeleton overlay still draws.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .. import io

__all__ = ["resolve_camera_files", "resolve_footage", "FrameSource"]

log = logging.getLogger("deeperfly")


def resolve_camera_files(
    info: dict[str, list[str]],
    results_dir: str | Path,
    footage_dir: str | Path | None = None,
) -> list[Path] | None:
    """Resolve one camera's footage files from its recorded ``{abs, rel}`` paths.

    Tries, in order: the absolute paths; the relative paths against
    ``results_dir``; the recorded file *names* under ``footage_dir`` (if given).
    Returns the first set whose files all exist, else ``None``.

    Parameters
    ----------
    info
        ``{"abs": [...], "rel": [...]}`` as stored by
        :meth:`~deeperfly.results.StageStore.write_pose2d`.
    results_dir
        The directory holding ``results.h5`` (anchors the relative paths).
    footage_dir
        An optional directory to search by file name as a last resort.

    Returns
    -------
    list of Path or None
        The resolved files, or ``None`` if none of the strategies found them.
    """
    abs_files = [Path(p) for p in info.get("abs", [])]
    if abs_files and all(p.exists() for p in abs_files):
        return abs_files
    rel_files = [Path(results_dir) / r for r in info.get("rel", [])]
    if rel_files and all(p.exists() for p in rel_files):
        return rel_files
    if footage_dir is not None:
        names = [Path(p).name for p in (info.get("abs") or info.get("rel") or [])]
        candidates = [Path(footage_dir) / n for n in names]
        if candidates and all(p.exists() for p in candidates):
            return candidates
    return None


def resolve_footage(
    footage: dict[str, dict[str, list[str]]] | None,
    results_dir: str | Path,
    footage_dir: str | Path | None = None,
) -> tuple[dict[str, list[Path]], list[str]]:
    """Resolve every camera's footage, splitting into found and missing.

    Parameters
    ----------
    footage
        ``camera_name -> {"abs": [...], "rel": [...]}`` (or ``None``).
    results_dir
        The directory holding ``results.h5``.
    footage_dir
        An optional directory to search by file name.

    Returns
    -------
    resolved, missing
        ``resolved`` maps each found camera to its files; ``missing`` lists the
        camera names whose footage could not be located.
    """
    resolved: dict[str, list[Path]] = {}
    missing: list[str] = []
    for name, info in (footage or {}).items():
        files = resolve_camera_files(info, results_dir, footage_dir)
        if files is None:
            missing.append(name)
        else:
            resolved[name] = files
    return resolved, missing


class FrameSource:
    """Per-camera frame decoding with a small LRU cache.

    Cameras absent from ``files_by_camera`` (footage not found) still answer
    :meth:`frame` with a black image of the recorded size, so the viewer can
    show the skeleton on a blank background instead of failing.
    """

    def __init__(
        self,
        files_by_camera: dict[str, list[Path]],
        image_sizes: dict[str, tuple[int, int]] | None = None,
        *,
        cache_size: int = 64,
    ):
        self._readers: dict[str, io.FrameReader] = {}
        self._counts: dict[str, int | None] = {}
        for name, files in files_by_camera.items():
            try:
                reader = io.open_reader(files)
            except Exception as exc:  # noqa: BLE001 -- footage may be unreadable
                log.warning("could not open footage for %s: %s", name, exc)
                continue
            self._readers[name] = reader
            self._counts[name] = reader.count()
        self._image_sizes = dict(image_sizes or {})
        self._cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._cache_size = cache_size

    @property
    def cameras(self) -> list[str]:
        """Names of cameras with an open reader (footage found)."""
        return list(self._readers)

    def n_frames(self) -> int | None:
        """The largest frame index every readable camera covers (``min`` count)."""
        counts = [c for c in self._counts.values() if c is not None]
        return min(counts) if counts else None

    def frame(self, name: str, idx: int) -> np.ndarray | None:
        """Decode camera ``name``'s frame ``idx`` (cached); blank if no footage.

        Returns ``None`` only when the camera has neither footage nor a recorded
        image size to synthesize a blank frame from.
        """
        if name not in self._readers:
            size = self._image_sizes.get(name)
            if size is None:
                return None
            height, width = size
            return np.zeros((int(height), int(width), 3), dtype=np.uint8)
        key = (name, idx)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        try:
            frame = np.asarray(self._readers[name][idx])
        except Exception as exc:  # noqa: BLE001 -- a bad frame shouldn't crash the UI
            log.warning("could not read %s frame %d: %s", name, idx, exc)
            return None
        self._cache[key] = frame
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return frame

    def close(self) -> None:
        """Close every open reader and drop the cache."""
        for reader in self._readers.values():
            try:
                reader.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("closing reader failed: %s", exc)
        self._readers.clear()
        self._cache.clear()
