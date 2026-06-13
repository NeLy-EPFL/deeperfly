"""The editor's data model: a :class:`PoseResult` plus a corrections overlay.

:class:`EditorState` is deliberately free of any Qt dependency -- it is the
testable core that the widgets drive. It exposes the *displayed* points
(corrected-over-original) for the current frame, applies 2D and 3D edits, and
holds the dirty/edit-mode flags.

The 3D edit is the interesting one. There is always a single internal 3D point
per keypoint; a drag re-estimates it and then refreshes every view's reprojection
with the same forward model used to draw it. Imperfect calibration means no single
3D point reprojects exactly onto all views at once, so a per-view point can be
*fixed* (finalized): a fixed view keeps its locked pixel and acts as a constraint.
Dropping a drag pins the dragged view there (it becomes fixed at the release
pixel), so the placed point stays put instead of snapping to the reprojection.

A per-view point can instead be *invisible* (obscured): a camera that genuinely
cannot see the keypoint is dropped from the triangulation entirely and its dot
just follows the reprojection (it cannot be dragged). Marking a view invisible
re-solves the 3D point from the remaining visible views, so a bad/occluded
observation stops dragging the estimate off. Invisible is mutually exclusive with
fixed (a point is at most one of fixed / invisible / plain).

On a drag we re-solve the 3D point by a constrained DLT
(:func:`deeperfly.triangulation.triangulate`) over the fixed views' locked pixels
plus the dragged view's cursor; with fewer than two such observations (the common
"nothing fixed yet" case) it falls back to the orthogonal projection of the old 3D
point onto the back-projection ray of the dragged pixel
(:func:`deeperfly.geometry.backproject_ray_one` +
:func:`deeperfly.geometry.closest_point_on_ray`), which lands the point exactly
under the cursor. Non-fixed views then follow the new 3D point's reprojection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from ..geometry import closest_point_on_ray
from ..results import PoseResult
from ..triangulation import triangulate
from .corrections import Corrections

__all__ = ["EditMode", "EditorState"]


class EditMode(str, Enum):
    """The interaction mode of the editor."""

    view = "view"  # read-only inspection
    edit_2d = "edit_2d"  # drag per-view 2D keypoints
    edit_3d = "edit_3d"  # drag reprojected 3D keypoints


@dataclass
class EditorState:
    """A loaded result plus its manual-corrections overlay and view state."""

    result: PoseResult
    corrections: Corrections
    frame: int = 0
    mode: EditMode = EditMode.view

    @classmethod
    def from_result(
        cls, result: PoseResult, corrections: Corrections | None = None
    ) -> EditorState:
        """Build a state for ``result``, with an empty overlay if none is given.

        A fresh overlay seeds the per-view "obscured" state from the detector: a
        NaN 2D detection means that camera did not see the keypoint, so the view
        starts obscured (dropped from triangulation, just following the
        reprojection) and the operator can drag it in to un-obscure it. The 3D
        point itself is left as the pipeline solved it (already triangulated from
        the finite views). A loaded sidecar keeps its own saved invisible mask.
        """
        if corrections is None:
            corrections = Corrections.empty(
                result.n_views, result.n_frames, cls._n_points(result)
            )
            corrections.pts2d_invisible = ~np.isfinite(result.pts2d).all(axis=-1)
        return cls(result=result, corrections=corrections)

    @staticmethod
    def _n_points(result: PoseResult) -> int:
        return int(result.pts2d.shape[2])

    # -- dimensions -----------------------------------------------------------

    @property
    def n_views(self) -> int:
        return self.result.n_views

    @property
    def n_frames(self) -> int:
        return self.result.n_frames

    @property
    def n_points(self) -> int:
        return self._n_points(self.result)

    @property
    def has_3d(self) -> bool:
        """Whether the result carries 3D points (enables :attr:`EditMode.edit_3d`)."""
        return self.result.pts3d is not None

    @property
    def camera_names(self) -> list[str]:
        return self.result.cameras.names

    @property
    def dirty(self) -> bool:
        return self.corrections.dirty

    def _resolve_frame(self, frame: int | None) -> int:
        return self.frame if frame is None else frame

    # -- displayed points (corrected over original) ---------------------------

    def display_pts2d(self, frame: int | None = None) -> Float[np.ndarray, "V P 2"]:
        """The 2D points to draw for ``frame``: corrections over the originals."""
        t = self._resolve_frame(frame)
        base = self.result.pts2d[:, t]
        ov = self.corrections.pts2d[:, t]
        mask = self.corrections.pts2d_edited[:, t]
        return np.where(mask[..., None], ov, base)

    def display_pts3d(
        self, frame: int | None = None
    ) -> Float[np.ndarray, "P 3"] | None:
        """The 3D points for ``frame`` (corrections over originals), or ``None``."""
        if self.result.pts3d is None:
            return None
        t = self._resolve_frame(frame)
        base = self.result.pts3d[t]
        ov = self.corrections.pts3d[t]
        mask = self.corrections.pts3d_edited[t]
        return np.where(mask[..., None], ov, base)

    def display_pts3d_projected(
        self, frame: int | None = None
    ) -> Float[np.ndarray, "V P 2"] | None:
        """The 3D points for ``frame`` reprojected into every view, or ``None``.

        Uses the same full forward model (:meth:`CameraGroup.project`) the
        overlays are drawn with, so a 3D edit lands exactly under the cursor.
        """
        pts3d = self.display_pts3d(frame)
        if pts3d is None:
            return None
        return np.asarray(self.result.cameras.project(pts3d))

    def display_pts2d_refine(
        self, frame: int | None = None
    ) -> Float[np.ndarray, "V P 2"] | None:
        """The per-view 2D drawn in Edit 3D: the 3D point reprojected into every
        view, with each *fixed* view overridden by its locked pixel, or ``None``.

        This is the "corrected 2D" result of a refined frame: non-fixed views are
        a single 3D point's reprojection while fixed views hold the operator's
        finalized pixels (which generally do not all agree with one 3D point).
        """
        proj = self.display_pts3d_projected(frame)
        if proj is None:
            return None
        t = self._resolve_frame(frame)
        fixed = self.corrections.pts2d_fixed[:, t]  # (V, P)
        locked = self.corrections.pts2d[:, t]  # (V, P, 2)
        return np.where(fixed[..., None], locked, proj)

    # -- edits ----------------------------------------------------------------

    def apply_2d_edit(
        self, view: int, point: int, xy, frame: int | None = None
    ) -> None:
        """Move ``point`` in ``view`` to pixel ``xy`` (independent per view)."""
        self.corrections.set_pts2d(view, self._resolve_frame(frame), point, xy)

    def apply_3d_edit(
        self, view: int, point: int, xy, frame: int | None = None, *, fix: bool = False
    ) -> Float[np.ndarray, "3"] | None:
        """Re-solve ``point``'s 3D location from a drag to pixel ``xy`` in ``view``.

        The 3D point is re-estimated by a constrained DLT over the *fixed* views'
        locked pixels plus the dragged view's cursor; non-fixed views then follow
        its reprojection. With fewer than two such observations (e.g. nothing is
        fixed yet) it falls back to the orthogonal projection of the old 3D point
        onto the back-projection ray of ``xy``, which lands the point exactly
        under the cursor (the original Edit 3D behavior).

        With ``fix=True`` (a drag *release*) the dragged view is finalized at
        ``xy``: it is pinned there as a locked constraint so it stays exactly
        where it was dropped instead of snapping to the reprojection. The live
        re-solve mid-drag uses ``fix=False`` so a view is only pinned on release
        (or if it was already fixed, in which case its lock follows the cursor).

        Dragging an *obscured* view un-obscures it (back to the normal state) and
        proceeds: the operator is placing it, so it rejoins the estimate.

        Returns the new 3D point, or ``None`` if there is no 3D point to move
        (no triangulation, or no usable constraint and the point is NaN here).

        Parameters
        ----------
        view
            The view index the user dragged in.
        point
            The skeleton point index being moved.
        xy
            The pixel the user dragged the point to, ``(2,)``.
        frame
            The frame to edit (defaults to the current frame).
        fix
            Whether to finalize (pin) the dragged view at ``xy`` -- set on a drag
            release so the dropped point persists; left ``False`` for the live
            mid-drag re-solve.
        """
        if self.result.pts3d is None:
            return None
        t = self._resolve_frame(frame)
        if self.corrections.pts2d_invisible[view, t, point]:
            # Dragging an obscured view un-obscures it: the operator is placing it,
            # so it re-enters the normal flow (and contributes to the re-solve below).
            self.corrections.set_invisible(view, t, point, False)
        xy = np.asarray(xy, dtype=float)
        fixed = self.corrections.pts2d_fixed[:, t, point]  # (V,)

        # Observations for the constrained DLT: each fixed view at its locked
        # pixel, plus the dragged view at the cursor (overriding if it is fixed).
        obs = np.full((self.n_views, 2), np.nan)
        obs[fixed] = self.corrections.pts2d[fixed, t, point]
        obs[view] = xy

        if int(np.isfinite(obs).all(axis=1).sum()) >= 2:
            x_new = np.asarray(
                triangulate(self.result.cameras, obs[:, None, :])[0], dtype=float
            )
        else:
            pts3d = self.display_pts3d(t)
            assert pts3d is not None
            x_old = pts3d[point]
            if not np.all(np.isfinite(x_old)):
                return None
            camera = list(self.result.cameras)[view]
            origin, direction = camera.backproject_ray(xy)
            x_new = np.asarray(
                closest_point_on_ray(
                    jnp.asarray(origin), jnp.asarray(direction), jnp.asarray(x_old)
                ),
                dtype=float,
            )
        if not np.all(np.isfinite(x_new)):
            return None
        self.corrections.set_pts3d(t, point, x_new)
        if fix or bool(fixed[view]):
            self.corrections.set_pts2d(view, t, point, xy, fixed=True)
        return x_new

    def toggle_fixed(
        self, view: int, point: int, frame: int | None = None
    ) -> bool | None:
        """Toggle whether ``point`` in ``view`` is finalized (a 3D constraint).

        Fixing snapshots the view's current displayed 2D as a locked pixel;
        unfixing drops it back to following the reprojection. Either way the 3D
        point is re-estimated from the (new) fixed set so the non-fixed views
        update. Returns the new fixed state, or ``None`` if there is no 3D point
        to refine or the point is not visible in this view.
        """
        if self.result.pts3d is None:
            return None
        t = self._resolve_frame(frame)
        cur2d = self.display_pts2d_refine(t)
        if cur2d is None:
            return None
        now_fixed = not bool(self.corrections.pts2d_fixed[view, t, point])
        if now_fixed:
            xy = cur2d[view, point]
            if not np.all(np.isfinite(xy)):
                return None  # cannot fix a point that is not visible in this view
            self.corrections.set_pts2d(view, t, point, xy, fixed=True)
        else:
            self.corrections.clear_2d(view, t, point)
        self._resolve_3d_from_fixed(point, t)
        return now_fixed

    def _resolve_3d_from_fixed(self, point: int, t: int) -> None:
        """Re-triangulate ``point``'s 3D location from its fixed views alone.

        A no-op below two fixed views (the 3D point keeps its current value).
        """
        fixed = self.corrections.pts2d_fixed[:, t, point]  # (V,)
        if int(fixed.sum()) < 2:
            return
        obs = np.full((self.n_views, 2), np.nan)
        obs[fixed] = self.corrections.pts2d[fixed, t, point]
        x_new = np.asarray(
            triangulate(self.result.cameras, obs[:, None, :])[0], dtype=float
        )
        if np.all(np.isfinite(x_new)):
            self.corrections.set_pts3d(t, point, x_new)

    def toggle_invisible(
        self, view: int, point: int, frame: int | None = None
    ) -> bool | None:
        """Toggle whether ``point`` in ``view`` is obscured (dropped from triangulation).

        An obscured view contributes nothing to the 3D point and simply follows its
        reprojection (it cannot be dragged); toggling re-solves the 3D from the
        remaining visible views so the estimate updates. Setting it clears any 2D
        edit / fixed flag for that view (invisible is mutually exclusive with fixed).
        Returns the new invisible state, or ``None`` if there is no 3D to refine.
        """
        if self.result.pts3d is None:
            return None
        t = self._resolve_frame(frame)
        now_invisible = not bool(self.corrections.pts2d_invisible[view, t, point])
        self.corrections.set_invisible(view, t, point, now_invisible)
        self._resolve_3d_from_visible(point, t)
        return now_invisible

    def _resolve_3d_from_visible(self, point: int, t: int) -> None:
        """Re-triangulate ``point``'s 3D location from its non-obscured views.

        With two or more fixed views the fixed pixels define the point (deferring
        to :meth:`_resolve_3d_from_fixed`, which already ignores the obscured,
        non-fixed views); otherwise the point is triangulated from every visible
        view's displayed 2D (the detector point unless edited/fixed), with the
        obscured views dropped. A no-op below two usable observations.
        """
        fixed = self.corrections.pts2d_fixed[:, t, point]  # (V,)
        if int(fixed.sum()) >= 2:
            self._resolve_3d_from_fixed(point, t)
            return
        invisible = self.corrections.pts2d_invisible[:, t, point]  # (V,)
        obs = self.display_pts2d(t)[:, point].astype(float)  # (V, 2)
        obs[invisible] = np.nan
        if int(np.isfinite(obs).all(axis=1).sum()) < 2:
            return
        x_new = np.asarray(
            triangulate(self.result.cameras, obs[:, None, :])[0], dtype=float
        )
        if np.all(np.isfinite(x_new)):
            self.corrections.set_pts3d(t, point, x_new)

    def reset_point(self, point: int, frame: int | None = None) -> None:
        """Drop every correction (all views' 2D, the fixed flags, the 3D) of ``point``."""
        t = self._resolve_frame(frame)
        for view in range(self.n_views):
            self.corrections.clear_2d(view, t, point)  # also clears the fixed flag
        self.corrections.clear_3d(t, point)

    def reset_point_view(self, view: int, point: int, frame: int | None = None) -> None:
        """Drop just ``view``'s 2D correction (and fixed flag) for ``point``.

        Unlike :meth:`reset_point`, the shared 3D point is left in place; if two or
        more views remain fixed it is re-solved from them so the other views still
        agree (a no-op below two fixed views). Use this to revert one view without
        discarding the work in the others.
        """
        t = self._resolve_frame(frame)
        self.corrections.clear_2d(view, t, point)  # also clears the fixed flag
        self._resolve_3d_from_fixed(point, t)

    def reset_frame(self, frame: int | None = None) -> None:
        """Drop every correction in ``frame`` -- all points, all views' 2D, the
        fixed/obscured flags, and 3D -- back to the pipeline's original pose."""
        t = self._resolve_frame(frame)
        self.corrections.clear_frame(t)
