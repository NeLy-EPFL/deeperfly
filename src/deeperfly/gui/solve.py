"""Deriving one keypoint's 3D from its per-view labels + predictions.

The editor's central operation: given, for one ``(frame, point)``, the operator's
GT pixels and the evidence behind every other view, produce the 3D point. This is where
the "2D is the source, 3D is derived" philosophy lives, and where the configurable
:class:`~deeperfly.config.AnnotationParams` solve policy is applied. Nothing here reads
the editor's **hidden** flag -- that decides which cells a training loss uses, and the
geometry has no opinion on it. Two entry points:

- :func:`solve_point_3d` -- the pure recompute (mass-confirm, navigation).
  Below two usable views it returns ``NaN`` (no fallback).

The ``gt_wins`` policy is built on one idea applied twice: **GT decides everything it has
an opinion about, and the remaining views supply only what it cannot determine.** A camera
fixes where a point sits across its optical axis and says nothing about distance along it,
so GT views leave a direction (or several) free, and that free part is where the
detections still belong. With one GT view the free part is the whole viewing ray, and
:func:`solve_depth_on_ray` refits a depth along it; with two or more it is whichever
direction the GT pair is worst at -- catastrophically so for two cameras that face each
other, whose viewing rays are nearly the same line -- and
:func:`solve_point_3d_stabilized` refits that. Neither needs to *ask* whether the geometry
is degenerate: weighting each observation by its own precision makes the answer come out
right at every angle, because along a direction the GT cannot see its weight is
identically zero however large it is.
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
from ..rig.triangulation import triangulate, triangulate_ransac

__all__ = [
    "solve_point_3d",
    "solve_point_3d_drag",
    "solve_depth_on_ray",
    "solve_point_3d_stabilized",
]


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


#: Central-difference step for :func:`_proj_and_jac`, in mm. The fly spans ~1.5 mm at
#: ~200 px/mm, so 1e-4 mm is ~0.02 px -- far above float64 cancellation and far below any
#: curvature in the projection. Checked against 1e-5: agreement to 2e-11 relative.
_JAC_H = 1e-4

#: ``(7, 3)`` offsets: the point itself, then ``+h`` and ``-h`` along each axis. Stacking
#: them means one projection call per iteration at one fixed ``(7, 1, 3)`` shape, so the
#: jitted projector specializes exactly once (see the module docstring on shapes).
_FD_OFFSETS = np.concatenate(
    [np.zeros((1, 3)), _JAC_H * np.eye(3), -_JAC_H * np.eye(3)]
)


def _proj_and_jac(cameras, x):
    """``(proj (V, 2), J (V, 2, 3))`` at ``x``: where it lands, and px per mm.

    ``J`` is ``d(project)/dx`` by central differences on the batch projector -- exact
    through distortion, because it differentiates the real projection rather than a
    pinhole approximation of it.
    """
    pts = (np.asarray(x, dtype=float)[None, :] + _FD_OFFSETS)[:, None, :]  # (7, 1, 3)
    p = np.asarray(cameras.project(pts), dtype=float)[:, :, 0]  # (V, 7, 2)
    jac = (p[:, 1:4] - p[:, 4:7]) / (2 * _JAC_H)  # (V, 3, 2), indexed [v, axis, comp]
    return p[:, 0], np.swapaxes(jac, 1, 2)


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


# -- two or more GT: a direction, not a whole point -----------------------------


def solve_point_3d_stabilized(
    cameras,
    x0: np.ndarray,
    gt_obs: np.ndarray,
    stab_obs: np.ndarray,
    ann: AnnotationParams,
    tri: TriangulationParams,
    *,
    max_iter: int = 12,
) -> np.ndarray:
    """Two or more GT views, with the rest supplying only what the GT cannot see.

    Two GT views do **not** determine a point equally well in every direction, and
    solving from them alone silently accepts whatever they are worst at. A camera fixes
    where a point sits *across* its optical axis and says nothing about distance *along*
    it: each view's ``J_v^T J_v`` is rank 2, with its own optical axis as the null
    direction. Two views facing each other therefore share that null direction -- for the
    standard rig's ``rm``/``lm`` pair, exactly: ``sum_v J_v^T J_v`` has eigenvalues
    ``[0, 8.7e4, 8.7e4]`` px^2/mm^2 and the zero one points along the shared axis. The
    two clicks are emphatic about four of the five things they could say and mute about
    the fifth, so a GT-only solve lets 0.5 px of click noise become 255 um mean / 372 um
    p90 of depth error, and the point lands 37 px off in the unlabelled views.

    So this does not choose between the operator and the detector; it weights each by how
    precise it is and lets the geometry decide who has a say in which direction::

        x_hat = argmin_x  sum_{v in GT}  |pi_v(x) - g_v|^2      / sigma_gt^2
                        + sum_{v in stab} rho_c(|pi_v(x) - s_v|) / sigma_stab^2

    That needs no conditioning test, no eigendecomposition and no threshold, because the
    arithmetic already does it. The GT's information in direction ``u`` is
    ``u^T H u / sigma_gt^2``: *exactly zero* along two anti-parallel rays' shared axis, so
    the stabilizers own that direction outright; and larger than theirs by
    ``(sigma_stab / sigma_gt)^2`` -- about 900x -- everywhere else, so they cannot budge
    it. Measured consequences of having no threshold: the correction grows smoothly with
    the geometry (0.1 um of motion at 90 degrees, 2.6 um at 165, 161 um at 180) with
    nothing for the operator to feel, and a well-conditioned pair is a genuine no-op
    (3.3 um and 0.30 px of GT residual, both unchanged).

    It also generalizes rather than special-cases: with one GT view ``H`` is rank 2 and
    free along the viewing ray, which is :func:`solve_depth_on_ray` (reproduced to 0.4 um
    mean / 1.6 um max); with three or more GT views ``H`` is full rank and large, and the
    stabilizers measurably cannot move the answer at all.

    ``rho_c`` is Huber at ``c = tri.ransac_threshold``, the same transition
    :func:`solve_depth_on_ray` uses and for the same measured reason -- a hard consensus
    throws away the information that separates "every view mildly wrong" from "two views
    lucky". Without it one 80 px stabilizer costs a weighted DLT most of its benefit
    (29 -> 69 um); with it the cost is 29 -> 32 um. ``sigma_stab`` reuses that same
    ``ransac_threshold`` as the detector's pixel scale, so the only new number is
    ``ann.gt_sigma_px``.

    Deterministic: a fixed iteration count with no convergence tolerance that could
    terminate differently, so re-deriving the same labels twice gives the same bits --
    which the derived-3D cache and the undo history require.

    Parameters
    ----------
    cameras
        The camera rig.
    x0
        ``(3,)`` the GT-only solution. It is the anchor and the start, so the answer
        degrades to today's behavior rather than to something unrelated.
    gt_obs
        ``(V, 2)`` GT pixels, ``NaN`` where the operator authored none. Distorted (raw)
        pixels: this path differentiates the real projection, so it never wants the
        linearized ``_undistort``.
    stab_obs
        ``(V, 2)`` the *independent* evidence that may fix the unconstrained direction --
        see :meth:`~deeperfly.gui.state.EditorState._point_obs` for why that is the
        detections rather than the instance's seeds.
    ann, tri
        ``ann.gt_sigma_px`` is the click precision; ``tri.ransac_threshold`` is both the
        Huber transition and the stabilizers' pixel scale.

    Returns
    -------
    np.ndarray
        The ``(3,)`` 3D point; ``x0`` unchanged when there is no usable stabilizer.
    """
    gt_obs = np.asarray(gt_obs, dtype=float)
    stab_obs = np.asarray(stab_obs, dtype=float)
    x = np.asarray(x0, dtype=float)
    gt_views = np.nonzero(np.isfinite(gt_obs).all(axis=-1))[0]
    stab_views = np.nonzero(np.isfinite(stab_obs).all(axis=-1))[0]
    if gt_views.size == 0 or stab_views.size == 0 or not np.all(np.isfinite(x)):
        return x

    c = float(tri.ransac_threshold)
    sigma_gt = max(float(ann.gt_sigma_px), 1e-6)
    w_gt = 1.0 / sigma_gt**2
    w_stab = 1.0 / max(c, 1e-6) ** 2
    eye = np.eye(3)

    for _ in range(max_iter):
        proj, jac = _proj_and_jac(cameras, x)
        r_gt = proj[gt_views] - gt_obs[gt_views]
        j_gt = jac[gt_views]
        normal = w_gt * np.einsum("vij,vik->jk", j_gt, j_gt)
        grad = w_gt * np.einsum("vij,vi->j", j_gt, r_gt)
        r_stab = proj[stab_views] - stab_obs[stab_views]
        j_stab = jac[stab_views]
        err = np.linalg.norm(r_stab, axis=-1)
        # Huber IRLS: a stabilizer that agrees keeps its full vote, one that is grossly
        # wrong keeps c/|r| of it -- down-weighted on its merits, never rejected.
        wt = w_stab * np.where(err <= c, 1.0, c / np.maximum(err, 1e-12))
        normal += np.einsum("v,vij,vik->jk", wt, j_stab, j_stab)
        grad += np.einsum("v,vij,vi->j", wt, j_stab, r_stab)
        # Rank deficiency is real here (one GT view plus one stabilizer is 4 equations
        # whose normal matrix can still be singular), and the ridge is scaled to the
        # matrix so it regularizes that case without biasing a healthy solve.
        ridge = 1e-12 * max(float(np.trace(normal)), 1e-30)
        step = -np.linalg.solve(normal + ridge * eye, grad)
        if not np.all(np.isfinite(step)):
            return np.asarray(x0, dtype=float)
        x = x + step
        if np.max(np.abs(step)) <= 1e-12:
            break
    return x if np.all(np.isfinite(x)) else np.asarray(x0, dtype=float)


# -- the two entry points -----------------------------------------------------


def solve_point_3d(
    cameras,
    gt_obs: np.ndarray,
    pred_obs: np.ndarray,
    conf: np.ndarray | None,
    ann: AnnotationParams,
    tri: TriangulationParams,
    stab_obs: np.ndarray | None = None,
) -> np.ndarray:
    """Recompute one point's 3D from its per-view labels + predictions.

    Parameters
    ----------
    cameras
        The camera rig.
    gt_obs
        ``(V, 2)`` GT pixels, ``NaN`` where the operator authored no GT.
    pred_obs
        ``(V, 2)`` the non-GT evidence (the instance's seeds, else the detections),
        ``NaN`` in a view that has none and in every view of an absent point -- the
        caller applies those vetoes (:meth:`EditorState._point_obs`).
    conf
        ``(V,)`` detector confidence, or ``None``.
    ann, tri
        The annotation solve policy and the shared triangulation method/thresholds.
    stab_obs
        ``(V, 2)`` the *independent* evidence for the ``gt_wins`` paths that fill in what
        the GT cannot determine -- the detections, which is not the same array as
        ``pred_obs`` once a frame has a ``seed_mode="triangulate"`` instance (see
        :meth:`EditorState._point_obs`). ``None`` falls back to ``pred_obs``, which keeps
        every existing caller and test working.

    Returns
    -------
    np.ndarray
        The ``(3,)`` 3D point, ``NaN`` if fewer than two usable views.
    """
    gt_obs = np.asarray(gt_obs, dtype=float)
    pred_obs = np.asarray(pred_obs, dtype=float)
    gt_raw, pred_raw = gt_obs, pred_obs  # the nonlinear paths want the real pixels
    if ann.undistort_before_solve:
        gt_obs = _undistort(cameras, gt_obs)
        pred_obs = _undistort(cameras, pred_obs)
    gt_mask = np.isfinite(gt_obs).all(axis=-1)
    pred_mask = np.isfinite(pred_obs).all(axis=-1) & ~gt_mask  # GT overrides per view
    n_gt = int(gt_mask.sum())
    policy = ann.solve_policy
    # The GT views' own pixels are the authored truth, never a stabilizer for it.
    stab_raw = pred_raw if stab_obs is None else np.asarray(stab_obs, dtype=float)
    stab_raw = np.where(gt_mask[:, None], np.nan, stab_raw)
    if not np.isfinite(stab_raw).all(axis=-1).any():
        # No independent observation anywhere for this point -- an all-contralateral
        # keypoint the detector never predicts. Fall back to the seeds, which is what
        # this path used before there was a distinction, rather than to no evidence.
        stab_raw = np.where(gt_mask[:, None], np.nan, pred_raw)

    if policy == "gt_wins":
        if n_gt >= ann.min_gt_for_exclusive:
            x0 = _dlt(cameras, np.where(gt_mask[:, None], gt_obs, np.nan))
            if not ann.gt_wins_keep_stabilizers:
                return x0  # GT alone: whatever the GT pair is worst at, it is worst at
            # Two GT views are mute about depth along a shared optical axis; the other
            # views are not, so they fill in that direction and nothing else.
            return solve_point_3d_stabilized(cameras, x0, gt_raw, stab_raw, ann, tri)
        if n_gt == 1:
            # One GT fixes the viewing ray, so what is left is a *depth*, and a depth can
            # be found by consensus rather than by averaging. This is the branch a plain
            # weighted DLT made non-robust: with a single GT the depth comes entirely from
            # the detections, so one bad peak carried 1/(V-1) of the answer -- and placing
            # a first GT would *remove* the RANSAC the zero-GT branch below enjoys.
            # It takes ``stab_raw`` for the same reason the two-GT branch does: a depth
            # read off reprojected seeds is the depth those seeds were made from.
            x = solve_depth_on_ray(
                cameras,
                int(np.nonzero(gt_mask)[0][0]),
                gt_raw[gt_mask][0],
                stab_raw,
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
