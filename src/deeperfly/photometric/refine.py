"""Refine the cross-side camera pose by chamfer-matching reprojected far-leg bones.

Solves for the single 6-DOF rigid transform ``T`` (see
:mod:`~deeperfly.photometric.objective`) that best pulls the reprojected *far*-side leg
bones onto the image leg pixels, encoded as a truncated distance transform per
(view, frame). A sparse keypoint-reprojection **anchor** (the existing detections,
including the front-camera bridge) is kept so ``T`` cannot overfit the noisy image term
and break the good near-side / bridge calibration. The Jacobian is finite-differenced
over the 6 parameters (cheap and robust for so few unknowns).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
from jaxtyping import Float
from scipy.optimize import OptimizeResult, least_squares

from ..cameras import CameraGroup
from .objective import (
    compensate_left,
    far_leg_bones,
    left_point_indices,
    project_np,
    rigid,
)

DEFAULT_LEFT = ("lf", "lm", "lh")
DEFAULT_RIGHT = ("rh", "rm", "rf")
DEFAULT_FRONT = ("f",)


@dataclass
class CrossSideRefinement:
    """Result of :func:`refine_extrinsics_photometric`."""

    delta: Float[np.ndarray, "6"]
    cameras: CameraGroup  # left cameras moved by T; right + front unchanged
    result: OptimizeResult
    n_chamfer: int  # kept far-bone samples
    n_anchor: int  # keypoint anchor observations
    chamfer_before: dict[str, float]  # per-view mean sampled DT (working px), init
    chamfer_after: dict[str, float]  # per-view mean sampled DT (working px), solved


def _cam_side(name, left, right, front):
    if name in left:
        return "left"
    if name in right:
        return "right"
    if name in front:
        return "front"
    return "other"


def build_problem(
    cameras: CameraGroup,
    pts3d: Float[np.ndarray, "F P 3"],
    dt_maps: dict[str, Float[np.ndarray, "F Hv Wv"]],
    scale: dict[str, float],
    skeleton,
    *,
    left=DEFAULT_LEFT,
    right=DEFAULT_RIGHT,
    front=DEFAULT_FRONT,
    samples_per_bone: int = 9,
    trunc_px: float = 20.0,
    gate_px: float | None = 9.0,
    reg: float = 3e-2,
    chamfer_scale: float = 10.0,
    pts2d_obs: dict[str, Float[np.ndarray, "F P 2"]] | None = None,
    conf: dict[str, Float[np.ndarray, "F P"]] | None = None,
    kpt_scale: float = 4.0,
    kpt_weight: float = 1.0,
):
    """Assemble the (chamfer + keypoint-anchor + Tikhonov) residual over ``delta``.

    Returns ``(eval_resid, keep)`` where ``eval_resid(delta, return_percam=False)`` gives
    the stacked residual (all blocks prescaled to O(1)) and ``keep`` is the gate mask over
    far-bone samples.
    """
    names = list(cameras.names)
    F, P = pts3d.shape[:2]
    # map_coordinates needs float32/64; accept float16 storage from build_leg_maps.
    dt_maps = {n: np.asarray(v, np.float32) for n, v in dt_maps.items()}
    ss = np.linspace(0.0, 1.0, samples_per_bone)
    left_pts = left_point_indices(skeleton)
    side = {n: _cam_side(n, left, right, front) for n in names}
    far_bones = {n: far_leg_bones(skeleton, side[n]) for n in names}
    cam = {
        n: (
            np.asarray(cameras[n].rvec),
            np.asarray(cameras[n].tvec),
            np.asarray(cameras[n].kmat),
            np.asarray(cameras[n].dist),
        )
        for n in names
    }
    use_kpt = pts2d_obs is not None and conf is not None

    def _proj_all(delta):
        moved = pts3d.copy()
        moved[:, left_pts] = rigid(pts3d[:, left_pts].reshape(-1, 3), delta).reshape(
            F, len(left_pts), 3
        )
        out = {}
        for n in names:
            rvec, tvec, kmat, dist = cam[n]
            if side[n] == "left":
                rvec, tvec = compensate_left(rvec, tvec, delta)
            out[n] = np.stack(
                [project_np(moved[f], rvec, tvec, kmat, dist) for f in range(F)]
            )
        return out

    def _cham(proj_all):
        for n in names:
            dtv, s = dt_maps[n], scale[n]
            for f in range(F):
                proj = proj_all[n][f]
                for a, b, w in far_bones[n]:
                    pa, pb = proj[a], proj[b]
                    if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                        yield n, None, w
                        continue
                    xy = (pa[None] * (1 - ss[:, None]) + pb[None] * ss[:, None]) * s
                    d = ndi.map_coordinates(
                        dtv[f],
                        [xy[:, 1], xy[:, 0]],
                        order=1,
                        mode="constant",
                        cval=trunc_px,
                    )
                    yield n, d, w

    keep = np.array(
        [
            (d is not None) and (w > 0) and (gate_px is None or np.median(d) <= gate_px)
            for n, d, w in _cham(_proj_all(np.zeros(6)))
        ]
    )

    anchor_mask = (
        {n: (np.isfinite(pts2d_obs[n]).all(-1) & (conf[n] > 0)) for n in names}
        if use_kpt
        else None
    )

    def eval_resid(delta, return_percam=False):
        proj_all = _proj_all(delta)
        res, percam = [], {}
        for i, (n, d, w) in enumerate(_cham(proj_all)):
            if not keep[i] or d is None:
                continue
            res.append(w * (d / scale[n]) / chamfer_scale)
            percam.setdefault(n, []).append(d)
        if return_percam:
            return {n: np.concatenate(v) for n, v in percam.items()}
        blocks = [np.concatenate(res) if res else np.zeros(0)]
        if use_kpt:
            for n in names:
                m = anchor_mask[n] & np.isfinite(proj_all[n]).all(-1)
                if not m.any():
                    continue
                diff = (proj_all[n] - pts2d_obs[n])[m]
                sw = np.sqrt(conf[n][m])[:, None]
                blocks.append((kpt_weight * sw * diff / kpt_scale).ravel())
        blocks.append(np.sqrt(reg) * delta)
        return np.concatenate(blocks)

    return eval_resid, keep


def refine_extrinsics_photometric(
    cameras: CameraGroup,
    pts3d: Float[np.ndarray, "F P 3"],
    dt_maps: dict[str, Float[np.ndarray, "F Hv Wv"]],
    scale: dict[str, float],
    skeleton,
    *,
    pts2d_obs=None,
    conf=None,
    f_scale: float = 1.0,
    max_nfev: int = 80,
    left=DEFAULT_LEFT,
    right=DEFAULT_RIGHT,
    front=DEFAULT_FRONT,
    **problem_kw,
) -> CrossSideRefinement:
    """Refine the left/right relative camera pose from far-leg image evidence.

    Parameters
    ----------
    cameras
        Input rig (e.g. the bundle-adjustment output).
    pts3d
        3D joints ``(F, P, 3)`` on the sampled frames (world coords, skeleton order).
    dt_maps, scale
        Per-view truncated distance transforms ``(F, Hv, Wv)`` and their raw->working
        pixel scale (see :func:`deeperfly.photometric.build_leg_maps`).
    skeleton
        Skeleton providing bones / limb sides / point names.
    pts2d_obs, conf
        Optional detected 2D keypoints ``(F, P, 2)`` (raw px) and confidences ``(F, P)``
        per view -- the sparse anchor that pins the bridge. Strongly recommended.

    Returns
    -------
    CrossSideRefinement
        The 6-vector ``delta``, the refined :class:`CameraGroup` (left cameras moved),
        the raw solver result, and before/after per-view chamfer diagnostics.
    """
    eval_resid, keep = build_problem(
        cameras,
        pts3d,
        dt_maps,
        scale,
        skeleton,
        left=left,
        right=right,
        front=front,
        pts2d_obs=pts2d_obs,
        conf=conf,
        **problem_kw,
    )
    before = {
        n: float(v.mean())
        for n, v in eval_resid(np.zeros(6), return_percam=True).items()
    }
    result = least_squares(
        eval_resid,
        np.zeros(6),
        jac="2-point",
        method="trf",
        loss="cauchy",
        f_scale=f_scale,
        max_nfev=max_nfev,
        diff_step=1e-4,
    )
    delta = result.x
    after = {
        n: float(v.mean()) for n, v in eval_resid(delta, return_percam=True).items()
    }

    # Build refined rig: left cameras moved by T; right + front unchanged.
    names = list(cameras.names)
    rvecs, tvecs = cameras.rvecs.copy(), cameras.tvecs.copy()
    for i, n in enumerate(names):
        if _cam_side(n, left, right, front) == "left":
            rvecs[i], tvecs[i] = compensate_left(rvecs[i], tvecs[i], delta)
    refined = CameraGroup.from_arrays(names, rvecs, tvecs, cameras.intrs, cameras.dists)

    n_anchor = 0
    if pts2d_obs is not None and conf is not None:
        n_anchor = int(
            sum(
                (np.isfinite(pts2d_obs[n]).all(-1) & (conf[n] > 0)).sum() for n in names
            )
        )
    return CrossSideRefinement(
        delta, refined, result, int(keep.sum()), n_anchor, before, after
    )
