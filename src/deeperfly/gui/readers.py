"""Locate each camera's footage and decode frames on demand for the viewer.

The footage paths recorded in ``results.h5`` (:meth:`StageStore.read_footage`)
are tried in order -- resolved-absolute, then relative to the ``results.h5``
directory, then a user-supplied directory (by file name) -- so a result moved or
copied still finds its videos. :class:`FrameSource` opens one
:class:`~deeperfly.io.base.FrameReader` per camera, reads through one open
:class:`~deeperfly.io.base.FrameCursor` each (so a step to the next frame costs a decode
rather than a re-open and a seek), and caches the last few decoded frames; a camera whose
footage cannot be found yields a black frame sized from the recorded ``image_sizes`` so
the skeleton overlay still draws.
"""

from __future__ import annotations

import logging
import threading
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
        A footage pointer (see :mod:`deeperfly.footage`), or a bare list of paths.
    results_dir
        The directory of the file the pointer came from (anchors ``rel`` and ``names``).
    footage_dir
        An optional directory to search by file name.

    Returns
    -------
    list of Path or None
        The resolved files, or ``None`` if none of the strategies found them.
    """
    from ..footage import resolve

    return resolve(info, results_dir, footage_dir)


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
    """Per-camera frame decoding, one open cursor each, behind a frame cache.

    Cameras absent from ``files_by_camera`` (footage not found) still answer
    :meth:`frame` with a black image of the recorded size, so the viewer can
    show the skeleton on a blank background instead of failing.

    Reads go through a per-camera :class:`~deeperfly.io.FrameCursor`, which keeps that
    camera's decoder open between frames. Measured on eight 1984x512 cameras served at
    once (decode plus JPEG encode, so a whole navigation): stepping to the next frame
    costs ~5 ms, against ~450 ms if each read re-opened and re-seeked the file. The
    cursors are opened on first use and dropped by :meth:`release_cache`, so a session
    nobody is looking at holds no decoder.

    Parameters
    ----------
    files_by_camera
        Resolved footage per camera (see :func:`resolve_footage`).
    image_sizes
        Recorded ``(height, width)`` per camera, used to synthesize a blank frame for a
        camera whose footage is missing.
    cache_bytes
        Memory budget for decoded frames, across all cameras. Budgeted in **bytes**, not
        frames, because a frame is 3 MiB of RGB or 1 MiB of gray depending on the footage
        and a frame count would mean something different for each.

        This is what makes stepping *backward* cheap. A video codec can only walk forward:
        going back one frame means seeking to the keyframe before it and decoding forward
        again -- some 200 frames on this footage, ~370 ms for the rig -- whereas the frames
        just behind the operator are ones this cache was handed moments ago, and answering
        from it costs ~2 ms. So the depth of free backward stepping is this budget divided
        by the rig's frame size: ~8 frames for eight monochrome 1984x512 cameras.
    gray_ok
        Whether monochrome footage may be handed back as ``(H, W)`` rather than three
        identical channels -- true by default because every consumer here branches on
        ``ndim``. It cuts the per-frame cost of serving a picture by about three quarters
        (0.8 ms against 3.2 ms: a 3 MB channel-reversing copy that disappears entirely,
        plus a one-channel JPEG) and, more importantly, thirds the size of a cached frame,
        which is what turns ~3 frames of free backward stepping into ~8. It saves almost
        no *bytes* on the wire -- JPEG already subsamples the flat chroma to nothing.
    """

    def __init__(
        self,
        files_by_camera: dict[str, list[Path]],
        image_sizes: dict[str, tuple[int, int]] | None = None,
        *,
        cache_bytes: int = 64 * 1024 * 1024,
        gray_ok: bool = True,
    ):
        self._readers: dict[str, io.FrameReader] = {}
        self._counts: dict[str, int | None] = {}
        self._files: dict[str, list[Path]] = {
            name: list(files) for name, files in files_by_camera.items()
        }
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
        self._cache_budget = cache_bytes
        self._cache_bytes = 0
        # The frame handlers run in a threadpool, so the cache and its running byte total
        # are shared mutable state. The lock covers only the bookkeeping -- never a decode
        # -- so the cameras of one navigation still decode in parallel.
        self._cache_lock = threading.Lock()
        self._gray_ok = gray_ok
        # Opened lazily, one per camera. Guarded because the frame handlers run in a
        # threadpool: without the lock two concurrent first requests for one camera would
        # each open a container and one would be dropped on the floor, still open.
        self._cursors: dict[str, io.FrameCursor] = {}
        self._cursor_lock = threading.Lock()

    @property
    def cameras(self) -> list[str]:
        """Names of cameras with an open reader (footage found)."""
        return list(self._readers)

    @property
    def footage_files(self) -> dict[str, list[Path]]:
        """The resolved footage backing each camera (empty for a blank-frame source).

        Identifies *which* pictures this source serves, which is what
        :func:`~deeperfly.gui.server._session_version` keys the browser cache on.
        """
        return {name: list(files) for name, files in self._files.items()}

    def n_frames(self) -> int | None:
        """The largest frame index every readable camera covers (``min`` count)."""
        counts = [c for c in self._counts.values() if c is not None]
        return min(counts) if counts else None

    def _cursor(self, name: str) -> io.FrameCursor | None:
        """Camera ``name``'s open cursor, opening one on first use.

        Lazy so that a source serving blank frames (or one nobody fetches from) opens
        nothing at all, and so a cursor dropped by :meth:`release_cache` simply comes
        back the next time that camera is asked for.
        """
        cursor = self._cursors.get(name)
        if cursor is not None:
            return cursor
        with self._cursor_lock:
            cursor = self._cursors.get(name)  # another thread may have just opened it
            if cursor is None:
                reader = self._readers.get(name)
                if reader is None:
                    return None
                cursor = reader.cursor(gray_ok=self._gray_ok)
                self._cursors[name] = cursor
            return cursor

    def _drop_cursor(self, name: str, cursor: io.FrameCursor) -> None:
        """Forget and close ``cursor`` -- unless it has already been replaced."""
        with self._cursor_lock:
            if self._cursors.get(name) is cursor:
                del self._cursors[name]
        try:
            cursor.close()
        except Exception as exc:  # noqa: BLE001
            log.debug("closing cursor for %s failed: %s", name, exc)

    def frame(self, name: str, idx: int) -> np.ndarray | None:
        """Decode camera ``name``'s frame ``idx`` (cached); blank if no footage.

        ``(H, W, 3)`` RGB, or ``(H, W)`` for monochrome footage unless the source was
        built with ``gray_ok=False``. Returns ``None`` only when the camera has neither
        footage nor a recorded image size to synthesize a blank frame from.
        """
        if name not in self._readers:
            size = self._image_sizes.get(name)
            if size is None:
                return None
            height, width = size
            return np.zeros((int(height), int(width), 3), dtype=np.uint8)
        key = (name, idx)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        frame = None
        cursor = self._cursor(name)
        if cursor is not None:
            try:
                frame = np.asarray(cursor.frame(idx))
            except Exception as exc:  # noqa: BLE001 -- fall back to a stateless read
                # A cursor can fail where the reader will not: `release_cache` closed its
                # container while this request was in flight (a recording switch), or a
                # seek went wrong on this particular file. Drop it -- the next request
                # opens a fresh one -- and answer from the reader, so the operator gets one
                # slow frame instead of a hole in the viewer.
                log.debug("cursor read of %s frame %d failed: %s", name, idx, exc)
                self._drop_cursor(name, cursor)
        if frame is None:
            try:
                frame = np.asarray(self._readers[name][idx])
            except Exception as exc:  # noqa: BLE001 -- a bad frame shouldn't crash the UI
                log.warning("could not read %s frame %d: %s", name, idx, exc)
                return None
        with self._cache_lock:
            if key not in self._cache:  # a concurrent request for the same frame won
                self._cache[key] = frame
                self._cache_bytes += frame.nbytes
            while self._cache_bytes > self._cache_budget and self._cache:
                self._cache_bytes -= self._cache.popitem(last=False)[1].nbytes
        return frame

    def release_cache(self) -> None:
        """Drop the decoded frames and close the cursors, keeping the readers.

        For a session the editor keeps but is no longer showing (the recording the
        operator switched away from, held for its unsaved labels). Both are pure cache,
        refilled on the way back, and both are what makes a retained session large: the
        decoded frames directly, and an open decoder through the reference frames and
        thread buffers FFmpeg keeps per stream -- a handful of megabytes per camera, held
        for every retained recording, for a picture nobody is looking at.

        The readers themselves stay: they hold no decoder and no OS handle, and are the
        part that took real time to resolve.

        Called while another thread may be inside :meth:`frame` on this source, which is why
        closing a cursor waits for a read in flight rather than freeing the container
        underneath it. A frame decoded by such a read lands in the cleared cache afterwards,
        which is harmless -- it is one picture, and the next release drops it.
        """
        with self._cache_lock:
            self._cache.clear()
            self._cache_bytes = 0
        with self._cursor_lock:
            cursors, self._cursors = self._cursors, {}
        for name, cursor in cursors.items():
            try:
                cursor.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("closing cursor for %s failed: %s", name, exc)

    def close(self) -> None:
        """Close every open cursor and reader, and drop the cache."""
        self.release_cache()
        for reader in self._readers.values():
            try:
                reader.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("closing reader failed: %s", exc)
        self._readers.clear()
