"""The manual-corrections sidecar (``corrections.h5``).

Corrections live in their own file next to ``results.h5`` so the pipeline's
output is never modified by the editor. Both an edited 2D point (per view) and
an edited 3D point (re-solved from a drag) are stored as dense arrays mirroring
the result's shapes, with NaN -- and an explicit boolean ``edited`` mask -- for
points the user has not touched. The mask is authoritative (so a deliberately
NaN-valued correction is still distinguishable from "not edited"), and lets a
single point be reset cleanly.

The layout (schema v3):

.. code-block:: text

    attrs["meta"]               json: {deeperfly_corrections_format_version, source, created_utc}
    pose2d_corrections/
        points                  (V, T, P, 2) edited 2D points (NaN where not edited)
        edited                  (V, T, P) bool
        fixed                   (V, T, P) bool  -- "finalized" per-view points used as
                                                   3D-refinement constraints (subset of edited)
        invisible               (V, T, P) bool  -- "obscured" per-view points dropped from
                                                   triangulation (disjoint from edited/fixed)
    pose3d_corrections/
        points3d                (T, P, 3) edited 3D points (NaN where not edited)
        edited                  (T, P) bool

The ``fixed`` mask is new in v2 and ``invisible`` in v3; older sidecars (without a
given mask) load with it all-False.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from jaxtyping import Bool, Float

__all__ = ["Corrections", "save_corrections", "load_corrections"]

CORRECTIONS_FORMAT_VERSION = 3


@dataclass
class Corrections:
    """In-memory overlay of manual edits on top of a :class:`PoseResult`.

    ``points`` arrays hold the edited values (NaN elsewhere); the ``edited``
    masks say which entries are real edits. ``pts2d_fixed`` marks the per-view
    2D points the operator has "finalized" -- they stay put under further edits
    and act as constraints when the 3D point is re-solved (always a subset of
    ``pts2d_edited``). ``pts2d_invisible`` marks the per-view points the operator
    has flagged "obscured" -- they are dropped from triangulation and follow the
    reprojection (mutually exclusive with ``pts2d_edited``/``pts2d_fixed``).
    ``dirty`` tracks unsaved in-memory changes (set on every edit, cleared by
    :func:`save_corrections`).
    """

    pts2d: Float[np.ndarray, "V T P 2"]
    pts2d_edited: Bool[np.ndarray, "V T P"]
    pts3d: Float[np.ndarray, "T P 3"]
    pts3d_edited: Bool[np.ndarray, "T P"]
    pts2d_fixed: Bool[np.ndarray, "V T P"]
    pts2d_invisible: Bool[np.ndarray, "V T P"]
    dirty: bool = field(default=False)

    @classmethod
    def empty(cls, n_views: int, n_frames: int, n_points: int) -> Corrections:
        """All-NaN overlays with empty masks for a ``(V, T, P)`` result."""
        return cls(
            pts2d=np.full((n_views, n_frames, n_points, 2), np.nan),
            pts2d_edited=np.zeros((n_views, n_frames, n_points), dtype=bool),
            pts3d=np.full((n_frames, n_points, 3), np.nan),
            pts3d_edited=np.zeros((n_frames, n_points), dtype=bool),
            pts2d_fixed=np.zeros((n_views, n_frames, n_points), dtype=bool),
            pts2d_invisible=np.zeros((n_views, n_frames, n_points), dtype=bool),
        )

    @property
    def any_edits(self) -> bool:
        """Whether any 2D or 3D point has been edited."""
        return bool(self.pts2d_edited.any() or self.pts3d_edited.any())

    def set_pts2d(
        self, view: int, frame: int, point: int, xy, *, fixed: bool = False
    ) -> None:
        """Record a 2D edit of ``point`` in ``view`` at ``frame``.

        With ``fixed=True`` the point is also marked finalized (a constraint for
        3D refinement); ``fixed=False`` leaves the existing fixed flag untouched.
        Placing a 2D point always clears the "obscured" flag -- an edited point is,
        by definition, visible.
        """
        self.pts2d[view, frame, point] = np.asarray(xy, dtype=float)
        self.pts2d_edited[view, frame, point] = True
        self.pts2d_invisible[view, frame, point] = False
        if fixed:
            self.pts2d_fixed[view, frame, point] = True
        self.dirty = True

    def set_invisible(self, view: int, frame: int, point: int, value: bool) -> None:
        """Flag ``point`` in ``view`` "obscured" (``value``), or clear the flag.

        An obscured point is dropped from triangulation and follows the
        reprojection, so it is mutually exclusive with an edited/fixed 2D: setting
        it drops any 2D edit and fixed flag for that view (back to the original).
        """
        self.pts2d_invisible[view, frame, point] = value
        if value:
            self.pts2d[view, frame, point] = np.nan
            self.pts2d_edited[view, frame, point] = False
            self.pts2d_fixed[view, frame, point] = False
        self.dirty = True

    def set_pts3d(self, frame: int, point: int, xyz) -> None:
        """Record a 3D edit of ``point`` at ``frame``."""
        self.pts3d[frame, point] = np.asarray(xyz, dtype=float)
        self.pts3d_edited[frame, point] = True
        self.dirty = True

    def clear_2d(self, view: int, frame: int, point: int) -> None:
        """Drop the 2D edit of ``point`` in ``view`` at ``frame`` (back to original)."""
        self.pts2d[view, frame, point] = np.nan
        self.pts2d_edited[view, frame, point] = False
        self.pts2d_fixed[view, frame, point] = False
        self.pts2d_invisible[view, frame, point] = False
        self.dirty = True

    def clear_3d(self, frame: int, point: int) -> None:
        """Drop the 3D edit of ``point`` at ``frame`` (back to original)."""
        self.pts3d[frame, point] = np.nan
        self.pts3d_edited[frame, point] = False
        self.dirty = True


def save_corrections(
    path: str | Path, corrections: Corrections, *, source: str | Path = ""
) -> None:
    """Write ``corrections`` to an HDF5 sidecar (overwriting ``path``).

    Clears ``corrections.dirty`` on success. ``results.h5`` is never touched.

    Parameters
    ----------
    path
        Destination ``corrections.h5`` path.
    corrections
        The overlay to persist.
    source
        Path to the ``results.h5`` these corrections apply to (stored in meta).
    """
    meta = {
        "deeperfly_corrections_format_version": CORRECTIONS_FORMAT_VERSION,
        "source": str(source),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with h5py.File(path, "w") as f:
        f.attrs["meta"] = json.dumps(meta)
        g2 = f.create_group("pose2d_corrections")
        g2.create_dataset("points", data=corrections.pts2d)
        g2.create_dataset("edited", data=corrections.pts2d_edited)
        g2.create_dataset("fixed", data=corrections.pts2d_fixed)
        g2.create_dataset("invisible", data=corrections.pts2d_invisible)
        g3 = f.create_group("pose3d_corrections")
        g3.create_dataset("points3d", data=corrections.pts3d)
        g3.create_dataset("edited", data=corrections.pts3d_edited)
    corrections.dirty = False


def load_corrections(
    path: str | Path, n_views: int, n_frames: int, n_points: int
) -> Corrections | None:
    """Read a ``corrections.h5`` sidecar, or ``None`` if it does not exist.

    Parameters
    ----------
    path
        Path to a ``corrections.h5`` written by :func:`save_corrections`.
    n_views, n_frames, n_points
        The current result's dimensions; the stored arrays must match.

    Returns
    -------
    Corrections or None
        The loaded overlay (``dirty`` is ``False``), or ``None`` if ``path`` is
        absent.

    Raises
    ------
    ValueError
        If the stored arrays do not match ``(n_views, n_frames, n_points)``
        (e.g. the corrections belong to a different result).
    """
    p = Path(path)
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        pts2d = np.asarray(f["pose2d_corrections/points"][()], dtype=float)  # type: ignore[index]
        pts2d_edited = np.asarray(f["pose2d_corrections/edited"][()], dtype=bool)  # type: ignore[index]
        pts3d = np.asarray(f["pose3d_corrections/points3d"][()], dtype=float)  # type: ignore[index]
        pts3d_edited = np.asarray(f["pose3d_corrections/edited"][()], dtype=bool)  # type: ignore[index]
        # "fixed" is new in schema v2 and "invisible" in v3; older sidecars load
        # with the missing mask all-False.
        if "pose2d_corrections/fixed" in f:
            pts2d_fixed = np.asarray(f["pose2d_corrections/fixed"][()], dtype=bool)  # type: ignore[index]
        else:
            pts2d_fixed = np.zeros(pts2d_edited.shape, dtype=bool)
        if "pose2d_corrections/invisible" in f:
            pts2d_invisible = np.asarray(f["pose2d_corrections/invisible"][()], dtype=bool)  # type: ignore[index]
        else:
            pts2d_invisible = np.zeros(pts2d_edited.shape, dtype=bool)
    want2d = (n_views, n_frames, n_points, 2)
    want3d = (n_frames, n_points, 3)
    if pts2d.shape != want2d or pts3d.shape != want3d:
        raise ValueError(
            f"{p} has corrections of shape 2D={pts2d.shape}, 3D={pts3d.shape}, "
            f"expected 2D={want2d}, 3D={want3d}; they belong to a different result"
        )
    return Corrections(
        pts2d=np.asarray(pts2d, dtype=float),
        pts2d_edited=pts2d_edited,
        pts3d=np.asarray(pts3d, dtype=float),
        pts3d_edited=pts3d_edited,
        pts2d_fixed=pts2d_fixed,
        pts2d_invisible=pts2d_invisible,
        dirty=False,
    )
