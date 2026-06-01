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
from jaxtyping import Bool, Float, Int

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


def merge_sources(
    remap: Int[np.ndarray, "N_old"],
    vis_mask: Bool[np.ndarray, "V N_old"],
    n_new: int,
) -> Int[np.ndarray, "V N_new"]:
    """Per-view source index for each merged point (see :func:`merge_points`).

    ``remap`` maps each old point index to its merged index. For a merged point
    fed by several old points (the left/right stripe pair) the source chosen for
    a view is the one that view can actually *see* (``vis_mask``); since the two
    stripe sides are visible to disjoint cameras this is unambiguous. Views that
    see no source keep an arbitrary (first) source -- the gathered value is NaN
    there anyway, since visibility was already applied to the observations.
    """
    n_view = vis_mask.shape[0]
    src = np.full((n_view, n_new), -1, dtype=np.int64)
    for old, new in enumerate(remap):
        first = src[:, new] == -1
        src[first, new] = old  # default: the first source feeding this point
        src[vis_mask[:, old], new] = old  # a source this view can see wins
    return src


def merge_points(
    arr: Float[np.ndarray, "V *pts"],
    src_for_dest: Int[np.ndarray, "V N_new"],
    *,
    axis: int = 2,
) -> Float[np.ndarray, "V *pts"]:
    """Collapse the point axis of ``arr`` using a per-view source index.

    ``arr`` is view-leading with its point axis at ``axis`` (2 for ``pts2d``
    ``(V,T,N,2)``, ``conf`` ``(V,T,N)`` and the candidate ``(V,T,N,K,...)``
    arrays). ``src_for_dest`` (from :func:`merge_sources`) gives, per view, the
    old point index to take for each merged point; the gather broadcasts over
    every other axis. This is how left/right stripe columns are fused into one.
    """
    arr = np.asarray(arr, dtype=float)
    ax = axis % arr.ndim
    n_view, n_new = src_for_dest.shape
    idx_shape = [1] * arr.ndim
    idx_shape[0] = n_view
    idx_shape[ax] = n_new
    idx = src_for_dest.reshape(idx_shape)
    idx = np.broadcast_to(idx, arr.shape[:ax] + (n_new,) + arr.shape[ax + 1 :])
    return np.take_along_axis(arr, idx, axis=ax)


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
