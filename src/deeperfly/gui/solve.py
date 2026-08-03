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

__all__ = ["solve_point_3d", "solve_point_3d_drag", "solve_depth_on_ray"]


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


# -- the one-GT case: a depth, not a triangulation ------------------------------


def _rays(cameras, views, pixels):
    """Unit-direction world rays ``(origin, direction)`` for ``pixels`` in ``views``.

    ``backproject_ray`` inverts projection exactly -- distortion included -- so the ray is
    the true set of world points that land on that pixel, independent of
    ``undistort_before_solve`` (which only governs the *linear* DLT paths).
    """
    cams = list(cameras)
    out = []
    for v, xy in zip(views, pixels):
        o, d = cams[int(v)].backproject_ray(np.asarray(xy, dtype=float))
        o = np.asarray(o, dtype=float)
        d = np.asarray(d, dtype=float)
        n = float(np.linalg.norm(d))
        out.append((o, d / n) if n > 0 else (o, d))
    return out


def _depth_from_rays(origin, direction, rays, weights=None):
    """The ``lambda`` minimizing the distance from ``origin + lambda*direction`` to ``rays``.

    Closed form. Each ray contributes the squared distance
    ``|A_v (origin + lambda*direction - o_v)|**2`` with ``A_v = I - d_v d_v^T`` the
    projector orthogonal to it; ``A_v`` is symmetric and idempotent, so differentiating the
    sum and setting it to zero gives

        lambda = -sum_v d^T A_v (origin - o_v) / sum_v d^T A_v d

    Minimizing 3D ray distance rather than reprojection error keeps this a one-line solve
    with no iteration, and it is the same algebraic-error compromise the DLT paths already
    make. ``None`` when every ray is parallel to ``direction`` (no depth information).
    """
    num = den = 0.0
    for i, (o_v, d_v) in enumerate(rays):
        wt = 1.0 if weights is None else float(weights[i])
        if wt <= 0.0:
            continue
        a = direction - d_v * float(direction @ d_v)  # A_v @ direction
        w = origin - o_v
        num += wt * float(a @ (w - d_v * float(w @ d_v)))  # a^T A_v w == d^T A_v w
        den += wt * float(a @ a)
    if den <= 1e-12:
        return None
    return -num / den


def _reproj_err(cameras, x, usable, pred_obs):
    """Per-usable-view reprojection error of the 3D point ``x``, in pixels."""
    proj = np.asarray(cameras.project(x[None, None, :]), dtype=float)[:, 0, 0]
    return np.linalg.norm(proj[usable] - pred_obs[usable], axis=-1)


def solve_depth_on_ray(
    cameras,
    gt_view: int,
    gt_xy,
    pred_obs: np.ndarray,
    tri: TriangulationParams,
    *,
    max_iter: int = 12,
) -> np.ndarray | None:
    """The point on the GT's viewing ray whose depth the detections agree on.

    With exactly one GT observation the problem is **one-dimensional**. The GT pixel fixes
    the viewing ray -- ``backproject_ray`` inverts projection exactly, distortion included --
    and the operator's pixel is not evidence about depth at all. So the only unknown is how
    far along that ray the keypoint sits, and the depth is refit *along* the ray, which puts
    the point exactly on it: it reprojects onto the operator's pixel to floating point,
    where a ``gt_weight``-weighted DLT only gets within ~0.1 px.

    The depth is estimated by **Huber IRLS** in pixel space, transitioning at
    ``tri.ransac_threshold``, rather than by a hard consensus. That choice is empirical, not
    stylistic. A consensus is the obvious tool here -- the minimal sample for a depth is one
    detection, so all candidates can be enumerated and scored -- but it fails in a regime
    that matters. Measured on the 7-camera fixture at 12 px detection noise with **no
    outliers at all**, enumerate-and-score came out 62-77% *worse* than the plain weighted
    DLT it replaces, whether scored by inlier count or by a truncated (MSAC) cost: once the
    noise approaches the threshold, truncation throws away exactly the information that
    separates "every view mildly wrong" from "two views lucky", and the fit locks onto a
    2-of-6 subset. Huber keeps every residual in the objective and only *down-weights* the
    large ones, so with nothing to reject it reduces to the least-squares fit -- no
    regression anywhere -- while still suppressing a gross outlier to ``c / |r|`` of its
    influence.

    Reusing ``ransac_threshold`` as the transition means this path adds no knob and inherits
    the scale ``[triangulation]`` already declares. Deterministic (a fixed number of
    reweighting steps from a fixed start), so re-deriving the same labels twice gives the
    same answer -- which the derived-3D cache depends on.

    Returns the ``(3,)`` point, or ``None`` when there is no usable detection to fix a depth
    (the caller then falls back to the mixed DLT).
    """
    pred_obs = np.asarray(pred_obs, dtype=float)
    usable = np.nonzero(np.isfinite(pred_obs).all(axis=-1))[0]
    if usable.size == 0:
        return None
    (origin, direction), *_ = _rays(cameras, [gt_view], [gt_xy])
    det_rays = _rays(cameras, usable, pred_obs[usable])
    c = float(tri.ransac_threshold)

    lam = _depth_from_rays(origin, direction, det_rays)  # least squares, then reweight
    if lam is None:
        return None
    for _ in range(max_iter):
        x = origin + lam * direction
        err = _reproj_err(cameras, x, usable, pred_obs)
        w = np.where(err <= c, 1.0, c / np.maximum(err, 1e-12))
        nxt = _depth_from_rays(origin, direction, det_rays, weights=w)
        if nxt is None:
            break
        if abs(nxt - lam) <= 1e-9 * max(1.0, abs(lam)):
            lam = nxt
            break
        lam = nxt
    x = origin + lam * direction
    return x if np.all(np.isfinite(x)) else None


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
        if n_gt == 1:
            # One GT fixes the viewing ray, so what is left is a *depth*, and a depth can
            # be found by consensus rather than by averaging. This is the branch a plain
            # weighted DLT made non-robust: with a single GT the depth comes entirely from
            # the detections, so one bad peak carried 1/(V-1) of the answer -- and placing
            # a first GT would *remove* the RANSAC the zero-GT branch below enjoys.
            x = solve_depth_on_ray(
                cameras,
                int(np.nonzero(gt_mask)[0][0]),
                gt_obs[gt_mask][0],
                pred_obs,
                tri,
            )
            if x is not None:
                return x
            # No candidate reached min_inliers: no consensus to be had, so fall through to
            # the non-robust mix, which at least uses every detection.
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
