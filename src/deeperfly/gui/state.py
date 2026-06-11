"""The editor's data model: a :class:`PoseResult` plus a corrections overlay.

:class:`EditorState` is deliberately free of any Qt dependency -- it is the
testable core that the widgets drive. It exposes the *displayed* points
(corrected-over-original) for the current frame, applies 2D and 3D edits, and
holds the dirty/edit-mode flags.

The 3D edit is the interesting one: dragging a point in one view to a pixel must
move the 3D point to the location that (1) reprojects exactly onto that pixel in
that view and (2) is closest to where the point was. That is the orthogonal
projection of the old 3D point onto the back-projection ray of the dragged pixel
(:func:`deeperfly.geometry.backproject_ray_one` +
:func:`deeperfly.geometry.closest_point_on_ray`), after which every other view's
reprojection is recomputed with the same forward model used to draw it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from ..geometry import closest_point_on_ray
from ..results import PoseResult
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
        """Build a state for ``result``, with an empty overlay if none is given."""
        if corrections is None:
            corrections = Corrections.empty(
                result.n_views, result.n_frames, cls._n_points(result)
            )
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

    # -- edits ----------------------------------------------------------------

    def apply_2d_edit(
        self, view: int, point: int, xy, frame: int | None = None
    ) -> None:
        """Move ``point`` in ``view`` to pixel ``xy`` (independent per view)."""
        self.corrections.set_pts2d(view, self._resolve_frame(frame), point, xy)

    def apply_3d_edit(
        self, view: int, point: int, xy, frame: int | None = None
    ) -> Float[np.ndarray, "3"] | None:
        """Re-solve ``point``'s 3D location from a drag to pixel ``xy`` in ``view``.

        Returns the new 3D point, or ``None`` if there is no 3D point to move
        (no triangulation, or the point is NaN at this frame). The returned
        point lies on the back-projection ray of ``xy`` through ``view`` (so it
        reprojects exactly onto ``xy`` there) and is the closest such point to
        the pre-drag 3D location.

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
        """
        if self.result.pts3d is None:
            return None
        t = self._resolve_frame(frame)
        pts3d = self.display_pts3d(t)
        assert pts3d is not None
        x_old = pts3d[point]
        if not np.all(np.isfinite(x_old)):
            return None
        camera = list(self.result.cameras)[view]
        origin, direction = camera.backproject_ray(np.asarray(xy, dtype=float))
        x_new = np.asarray(
            closest_point_on_ray(
                jnp.asarray(origin), jnp.asarray(direction), jnp.asarray(x_old)
            ),
            dtype=float,
        )
        self.corrections.set_pts3d(t, point, x_new)
        return x_new

    def reset_point(self, point: int, frame: int | None = None) -> None:
        """Drop every correction (all views' 2D and the 3D) of ``point`` at ``frame``."""
        t = self._resolve_frame(frame)
        for view in range(self.n_views):
            self.corrections.clear_2d(view, t, point)
        self.corrections.clear_3d(t, point)
