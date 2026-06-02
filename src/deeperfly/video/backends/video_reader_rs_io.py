"""video_reader-rs backend: a fast Rust/FFmpeg CPU decoder.

``video-reader-rs`` (https://github.com/gcanat/video_reader-rs, imported as
``video_reader``) decodes on multiple threads and exposes native random access
via ``get_batch``. Output is RGB ``(N, H, W, 3)`` NumPy. Both the functional API
and the newer ``PyVideoReader`` class are supported.

The wheel bundles its own FFmpeg as a private ``video_reader_rs.libs/`` (the
auditwheel pattern), but those bundled libs carry no RUNPATH back to that
directory, so the loader can't resolve their *transitive* dependencies (e.g.
``libavcodec`` -> ``libswresample``) and ``import video_reader`` dies with
``libswresample-...: cannot open shared object file``. :func:`_preload_ffmpeg`
``dlopen``s the bundled libs by absolute path before the import, the same trick
:mod:`..torchcodec_io` uses for NVIDIA NPP.
"""

from __future__ import annotations

import numpy as np

from ..base import ReaderBackend, register_reader, require_cpu

_ffmpeg_preloaded = False


def _preload_ffmpeg() -> None:
    """Pre-``dlopen`` video_reader-rs's bundled FFmpeg libs so its extension imports.

    The bundled libs depend on one another but expose no inter-lib RUNPATH, so a
    single pass can hit a not-yet-loaded dependency; retry (RTLD_GLOBAL, absolute
    path) until the unresolved set stops shrinking. No-op if the package or its
    private ``*.libs`` directory isn't present, and libs that never load (a
    genuinely missing system dep, off the decode path) are simply skipped.
    """
    global _ffmpeg_preloaded
    if _ffmpeg_preloaded:
        return
    import ctypes
    import glob
    import os
    import sysconfig

    libs: list[str] = []
    for site in {sysconfig.get_paths()[k] for k in ("purelib", "platlib")}:
        libs += glob.glob(os.path.join(site, "video_reader*.libs", "*.so*"))
    pending = libs
    while pending:
        unresolved = []
        for lib in pending:
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                unresolved.append(lib)
        if len(unresolved) == len(pending):  # no progress -> the rest are unloadable
            break
        pending = unresolved
    _ffmpeg_preloaded = True


def _load():
    """Import ``video_reader`` after preloading its bundled FFmpeg libraries."""
    _preload_ffmpeg()
    import video_reader

    return video_reader


def _reader(vr, path):
    """A ``PyVideoReader`` instance when available, else ``None`` (functional API)."""
    cls = getattr(vr, "PyVideoReader", None)
    return cls(str(path)) if cls is not None else None


@register_reader
class VideoReaderRsReader(ReaderBackend):
    name = "video_reader_rs"
    requires = ("video_reader",)
    supports_gpu = False
    supports_seek = True

    @classmethod
    def is_available(cls) -> bool:
        # The package can be importable as a name while its bundled FFmpeg shared
        # libraries fail to load, so the shallow find_spec check passes but any real
        # use crashes. Probe a true import (after the preload that makes those libs
        # resolvable) so a still-broken install is neither advertised nor selected.
        try:
            _load()
        except Exception:
            return False
        return True

    @staticmethod
    def _read_sequential(path, device, start, stop, step):
        require_cpu(device, "video_reader_rs")
        vr = _load()

        reader = _reader(vr, path)
        if reader is not None:
            arr = reader.decode(start_frame=start, end_frame=stop)
        else:
            arr = vr.decode(str(path), start_frame=start, end_frame=stop)
        arr = np.asarray(arr)
        return arr[::step] if step != 1 else arr

    @classmethod
    def _read_indices(cls, path, device, indices):
        require_cpu(device, "video_reader_rs")
        vr = _load()

        reader = _reader(vr, path)
        if reader is not None and hasattr(reader, "get_batch"):
            return np.asarray(reader.get_batch(list(indices)))
        if hasattr(vr, "get_batch"):
            return np.asarray(vr.get_batch(str(path), list(indices)))
        return super()._read_indices(path, device, indices)  # decode-and-gather
