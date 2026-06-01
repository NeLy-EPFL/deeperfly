"""Skeleton-aware triangulation helpers over a :class:`CameraGroup`.

Thin, NumPy-facing wrappers around the geometry already provided by
:mod:`deeperfly.geometry` and :class:`deeperfly.cameras.CameraGroup`. The single
contract with the geometry layer is the **NaN convention**: a 2D observation of
``NaN`` means "this camera did not (or cannot) see this point". Both
:func:`deeperfly.geometry.triangulate_dlt` and the bundle-adjustment residual
builder already honor it, so visibility is expressed purely as NaNs -- no
separate mask array travels downstream.

All functions use the **view-leading** layout shared with the geometry module:
``pts2d`` has shape ``(V, *pts, 2)`` (e.g. ``(V, N, 2)`` for one frame or
``(V, T, N, 2)`` for a sequence) and triangulated points come back as
``(*pts, 3)``.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Float

from .cameras import CameraGroup
from .skeleton import Skeleton


def apply_visibility(
    pts2d: Float[np.ndarray, "V *pts 2"],
    skeleton: Skeleton,
    camera_names: list[str],
) -> Float[np.ndarray, "V *pts 2"]:
    """Return a copy of ``pts2d`` with invisible (camera, point) entries NaN'd.

    ``camera_names`` labels the leading view axis of ``pts2d`` and is matched
    against :attr:`Skeleton.visibility`. Cameras unknown to the skeleton keep
    all their points (see :meth:`Skeleton.visibility_mask`).
    """
    pts2d = np.array(pts2d, dtype=float)
    mask = skeleton.visibility_mask(camera_names)  # (V, N)
    n_view, n_pts = mask.shape
    if pts2d.shape[0] != n_view:
        raise ValueError(
            f"pts2d has {pts2d.shape[0]} views but {n_view} camera names given"
        )
    if pts2d.shape[-2] != n_pts:
        raise ValueError(f"pts2d has {pts2d.shape[-2]} points but skeleton has {n_pts}")
    # Broadcast (V, N) over any middle (e.g. time) axes -> (V, *pts).
    n_mid = pts2d.ndim - 3
    m = mask.reshape((n_view, *([1] * n_mid), n_pts))
    return np.where(m[..., None], pts2d, np.nan)


def triangulate(
    cameras: CameraGroup,
    pts2d: Float[np.ndarray, "V *pts 2"],
) -> Float[np.ndarray, "*pts 3"]:
    """Triangulate 3D points from 2D observations (NaN-aware DLT).

    Points seen by fewer than two cameras come back as ``NaN``. This simply
    forwards to :meth:`CameraGroup.triangulate`; it exists as the pipeline's
    named entry point alongside :func:`apply_visibility` and
    :func:`reprojection_error`.
    """
    return cameras.triangulate(pts2d)


def reprojection_error(
    cameras: CameraGroup,
    pts3d: Float[np.ndarray, "*pts 3"],
    pts2d: Float[np.ndarray, "V *pts 2"],
) -> Float[np.ndarray, "V *pts"]:
    """Per-(view, point) reprojection error in pixels.

    Projects ``pts3d`` through every camera and takes the Euclidean distance to
    ``pts2d``. Entries are ``NaN`` wherever the observation or the 3D point is
    ``NaN`` (unobserved / un-triangulated), so callers can ignore them with
    ``np.nanmean`` / ``np.nanmax``.
    """
    proj = cameras.project(np.asarray(pts3d))  # (V, *pts, 2)
    return np.linalg.norm(proj - np.asarray(pts2d), axis=-1)
