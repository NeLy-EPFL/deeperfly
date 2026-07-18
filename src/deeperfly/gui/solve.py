"""Deriving one keypoint's 3D from its per-view labels + predictions.

The editor's central operation: given, for one ``(frame, point)``, the operator's
GT pixels, the detector's predictions, and which views are occluded, produce the 3D
point. This is where the "2D is the source, 3D is derived" philosophy lives, and
where the configurable :class:`~deeperfly.config.AnnotationParams` solve policy is
applied. Two entry points:

- :func:`solve_point_3d` -- the pure recompute (mass-confirm, occlude, navigation).
  Below two usable views it returns ``NaN`` (no fallback).
- :func:`solve_point_3d_drag` -- the interactive drag solve. It treats the dragged
  view as a fresh GT constraint and, below two usable views, back-projects the
  cursor and slides the prior 3D onto that ray so the point lands under the mouse.
  It always uses the cheap weighted-DLT / ray path regardless of ``solve_policy`` --
  the expensive consensus runs only on the settle (see the redesign doc §3.5).

Every triangulation is done one point at a time at the fixed ``(V, 1, 2)`` shape the
rest of the GUI uses, so the jitted DLT never recompiles per interaction.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..config import AnnotationParams, TriangulationParams
from ..geometry import closest_point_on_ray, undistort_one
from ..triangulation import triangulate, triangulate_ransac

__all__ = ["solve_point_3d", "solve_point_3d_drag"]


# -- low-level triangulation over one point -----------------------------------


def _dlt(cameras, obs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Plain (optionally weighted) DLT for one point. ``obs`` ``(V, 2)`` -> ``(3,)``."""
    w = None if weights is None else np.asarray(weights, dtype=float)[:, None]
    return np.asarray(triangulate(cameras, obs[:, None, :], w)[0], dtype=float)


def _configured(
    cameras, obs: np.ndarray, tri: TriangulationParams, conf: np.ndarray | None
) -> np.ndarray:
    """Triangulate one point with the *batch* method + thresholds (run parity).

    Uses the same ``[triangulation]`` estimator the pipeline used, so a point with no
    GT re-solves to the run's cached 3D. Confidence weighting follows the batch
    ``weigh_by_confidence`` (not the annotation prediction weight).
    """
    weights = conf if (tri.weigh_by_confidence and conf is not None) else None
    if tri.method == "ransac":
        w = None if weights is None else np.asarray(weights, dtype=float)[:, None]
        pts3d, _ = triangulate_ransac(
            cameras,
            obs[:, None, :],
            threshold=tri.ransac_threshold,
            min_inliers=tri.min_inliers,
            weights=w,
        )
        return np.asarray(pts3d[0], dtype=float)
    return _dlt(cameras, obs, weights)


# -- helpers ------------------------------------------------------------------


def _undistort(cameras, obs: np.ndarray) -> np.ndarray:
    """Map each finite pixel to its linear-pinhole (undistorted) pixel.

    A no-op for cameras without distortion coefficients. Keeps the linear DLT
    consistent with the distorted image the operator clicked in.
    """
    out = np.array(obs, dtype=float, copy=True)
    for v, cam in enumerate(cameras):
        xy = obs[v]
        if not np.all(np.isfinite(xy)):
            continue
        dist = np.asarray(cam.dist, dtype=float).reshape(-1)
        if dist.size == 0:
            continue
        fx, fy, cx, cy = (float(x) for x in np.asarray(cam.intr, dtype=float))
        xn = jnp.asarray([(xy[0] - cx) / fx, (xy[1] - cy) / fy])
        xu = np.asarray(undistort_one(xn, jnp.asarray(dist)), dtype=float)
        out[v] = [xu[0] * fx + cx, xu[1] * fy + cy]
    return out


def _pred_weight(
    ann: AnnotationParams, conf: np.ndarray | None, n_views: int
) -> np.ndarray:
    """Per-view weight for prediction rows in a GT-present weighted DLT."""
    pw = ann.prediction_weight
    if pw == "confidence" and conf is not None:
        return np.asarray(conf, dtype=float)
    if isinstance(pw, (int, float)) and not isinstance(pw, bool):
        return np.full(n_views, float(pw))
    return np.ones(n_views)  # "uniform" (and the confidence-without-conf fallback)


def _mix(
    gt_obs: np.ndarray,
    gt_mask: np.ndarray,
    pred_obs: np.ndarray,
    pred_mask: np.ndarray,
    conf: np.ndarray | None,
    ann: AnnotationParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the mixed ``(obs, weights)`` for a GT-present weighted DLT.

    GT views carry ``gt_weight``; prediction-only views carry the annotation
    prediction weight; unused views get weight 0 (dropped by the DLT).
    """
    n_views = gt_obs.shape[0]
    obs = np.where(
        gt_mask[:, None], gt_obs, np.where(pred_mask[:, None], pred_obs, np.nan)
    )
    weights = np.zeros(n_views)
    weights[gt_mask] = ann.gt_weight
    weights[pred_mask] = _pred_weight(ann, conf, n_views)[pred_mask]
    return obs, weights


# -- the two entry points -----------------------------------------------------


def solve_point_3d(
    cameras,
    gt_obs: np.ndarray,
    pred_obs: np.ndarray,
    conf: np.ndarray | None,
    ann: AnnotationParams,
    tri: TriangulationParams,
) -> np.ndarray:
    """Recompute one point's 3D from its per-view labels + predictions.

    Parameters
    ----------
    cameras
        The camera rig.
    gt_obs
        ``(V, 2)`` GT pixels, ``NaN`` where the operator authored no GT.
    pred_obs
        ``(V, 2)`` detector predictions, ``NaN`` where absent **or occluded** (the
        caller NaNs out occluded/absent views; occluded views must be NaN in both).
    conf
        ``(V,)`` detector confidence, or ``None``.
    ann, tri
        The annotation solve policy and the shared triangulation method/thresholds.

    Returns
    -------
    np.ndarray
        The ``(3,)`` 3D point, ``NaN`` if fewer than two usable views.
    """
    gt_obs = np.asarray(gt_obs, dtype=float)
    pred_obs = np.asarray(pred_obs, dtype=float)
    if ann.undistort_before_solve:
        gt_obs = _undistort(cameras, gt_obs)
        pred_obs = _undistort(cameras, pred_obs)
    gt_mask = np.isfinite(gt_obs).all(axis=-1)
    pred_mask = np.isfinite(pred_obs).all(axis=-1) & ~gt_mask  # GT overrides per view
    n_gt = int(gt_mask.sum())
    policy = ann.solve_policy

    if policy == "gt_wins":
        if n_gt >= ann.min_gt_for_exclusive and not ann.gt_wins_keep_stabilizers:
            obs = np.where(gt_mask[:, None], gt_obs, np.nan)
            return _dlt(cameras, obs)  # GT alone
        if n_gt >= 1:
            obs, weights = _mix(gt_obs, gt_mask, pred_obs, pred_mask, conf, ann)
            return _dlt(cameras, obs, weights)  # GT hard-weighted, predictions fill
        obs = np.where(pred_mask[:, None], pred_obs, np.nan)
        return _configured(cameras, obs, tri, conf)  # no GT: run-parity solve

    if policy == "weighted_blend":
        if n_gt == 0:
            obs = np.where(pred_mask[:, None], pred_obs, np.nan)
            return _configured(cameras, obs, tri, conf)
        obs, weights = _mix(gt_obs, gt_mask, pred_obs, pred_mask, conf, ann)
        return _dlt(cameras, obs, weights)

    if policy == "equal_weight":
        obs = np.where(
            gt_mask[:, None], gt_obs, np.where(pred_mask[:, None], pred_obs, np.nan)
        )
        if tri.method == "ransac":
            w = (
                np.asarray(conf, dtype=float)[:, None]
                if (tri.weigh_by_confidence and conf is not None)
                else None
            )
            pts3d, inliers = triangulate_ransac(
                cameras,
                obs[:, None, :],
                threshold=tri.ransac_threshold,
                min_inliers=tri.min_inliers,
                weights=w,
            )
            if ann.equal_weight_protect_gt and n_gt:
                # A prediction consensus must not vote a human GT out of its own solve:
                # force GT views back into the inlier set and refit from it.
                keep = inliers[:, 0] | gt_mask
                refit = np.where(keep[:, None], obs, np.nan)
                if int(np.isfinite(refit).all(axis=-1).sum()) >= 2:
                    return _dlt(cameras, refit)
            return np.asarray(pts3d[0], dtype=float)
        return _dlt(cameras, obs)

    raise ValueError(f"unknown solve_policy {policy!r}")


def solve_point_3d_drag(
    cameras,
    gt_obs: np.ndarray,
    pred_obs: np.ndarray,
    conf: np.ndarray | None,
    dragged_view: int,
    dragged_xy,
    prior_xyz: np.ndarray | None,
    ann: AnnotationParams,
) -> np.ndarray | None:
    """Cheap interactive drag solve; the dragged view becomes a GT constraint.

    With two or more usable views it is a GT-weighted DLT (so the point tracks the
    labelled views); with fewer it back-projects ``dragged_xy`` and slides
    ``prior_xyz`` the least distance onto that ray, landing the point exactly under
    the cursor. Returns the new ``(3,)`` point, or ``None`` when there is nothing to
    move (no prior 3D and too few views). Independent of ``solve_policy`` -- the
    expensive consensus is reserved for the settle recompute.
    """
    gt_obs = np.array(gt_obs, dtype=float, copy=True)
    gt_obs[dragged_view] = np.asarray(dragged_xy, dtype=float)
    pred_obs = np.asarray(pred_obs, dtype=float)
    gt_mask = np.isfinite(gt_obs).all(axis=-1)
    pred_mask = np.isfinite(pred_obs).all(axis=-1) & ~gt_mask
    obs, weights = _mix(gt_obs, gt_mask, pred_obs, pred_mask, conf, ann)
    solve_obs = _undistort(cameras, obs) if ann.undistort_before_solve else obs
    if int(np.isfinite(solve_obs).all(axis=-1).sum()) >= 2:
        x_new = _dlt(cameras, solve_obs, weights)
        if np.all(np.isfinite(x_new)):
            return x_new
    # Fewer than two usable views: slide the prior 3D onto the dragged ray.
    if prior_xyz is None or not np.all(np.isfinite(prior_xyz)):
        return None
    camera = list(cameras)[dragged_view]
    origin, direction = camera.backproject_ray(np.asarray(dragged_xy, dtype=float))
    x_new = np.asarray(
        closest_point_on_ray(
            jnp.asarray(origin), jnp.asarray(direction), jnp.asarray(prior_xyz)
        ),
        dtype=float,
    )
    return x_new if np.all(np.isfinite(x_new)) else None
