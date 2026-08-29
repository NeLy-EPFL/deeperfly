"""The result file: one self-contained HDF5 per recording.

:class:`~deeperfly.results.core.PoseResult` is the finished pose -- points, cameras and
the skeleton that names them -- and :class:`~deeperfly.results.core.StageStore` is the
staged store behind it, which every pipeline stage writes into and reads back so a rerun
can reuse what has not changed. :func:`~deeperfly.results.core.repack` rewrites an
existing file in the current schema without recomputing anything.
"""

from __future__ import annotations

from . import core
from .core import (
    FORMAT_VERSION,
    READABLE_VERSIONS,
    PoseResult,
    StageStore,
    repack,
    stored_version,
)

__all__ = [
    "core",
    "PoseResult",
    "StageStore",
    "repack",
    "stored_version",
    "FORMAT_VERSION",
    "READABLE_VERSIONS",
]
