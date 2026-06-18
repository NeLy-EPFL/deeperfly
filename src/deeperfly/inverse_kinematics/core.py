"""The IK solver: bounded per-frame least squares over each leg's joint angles.

For each leg the angles are fit frame by frame with
:func:`scipy.optimize.least_squares` (TRF, box-bounded by the joint limits) against
an analytic Jacobian from JAX -- the same JAX-Jacobian-into-scipy scheme bundle
adjustment uses. The residual is the leg's forward-kinematic joint positions minus
the measured ones in the leg-local frame, with unobserved joints masked out. Each
frame is seeded from the previous frame's solution for temporal continuity (the
first frame starts from a mid-range bent posture, away from the straight-leg
singularity).

The head/antenna angles are not a serial chain; :func:`solve_head` reads them off
the measured antenna-tip direction (the reference's vector method).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import least_squares

from .align import Alignment, to_local, to_world
from .articulation import Chain
from .kinematics import make_chain_kernels, make_leg_kernels
from .template import HeadSpec, LegChain

log = logging.getLogger("deeperfly")

__all__ = ["solve_leg", "solve_head", "solve_chain"]

#: A joint must move under at least this many valid observations for a useful fit.
_MIN_VALID_JOINTS = 2


def solve_leg(
    pts3d: np.ndarray,
    index: dict[str, int],
    chain: LegChain,
    align: Alignment,
    *,
    max_nfev: int = 100,
    loss: str = "linear",
    f_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one leg's joint angles over every frame.

    Parameters
    ----------
    pts3d
        The 3D pose ``(T, P, 3)`` in world coordinates (NaN for missing points).
    index
        ``point_name -> column`` map into ``pts3d`` (the skeleton order).
    chain
        The leg's kinematic chain (joints, DOF axes, bounds).
    align
        The body alignment (origin + segment lengths for this leg).
    max_nfev, loss, f_scale
        Forwarded to :func:`scipy.optimize.least_squares`.

    Returns
    -------
    angles : np.ndarray
        ``(T, D)`` fitted joint angles in radians (NaN for un-fittable frames).
    model_world : np.ndarray
        ``(T, J, 3)`` forward-kinematic joint positions in world coordinates.
    """
    n_frames = pts3d.shape[0]
    dof_counts = chain.dof_counts
    n_dofs = sum(dof_counts)
    n_joints = len(chain.joints)
    fk, jac = make_leg_kernels(dof_counts)

    axes = np.asarray(chain.axes, dtype=float)
    lo, hi = chain.bounds
    seglens = np.asarray(align.seglens[chain.name], dtype=float)
    origin = align.leg_origin[chain.name]
    r_body = align.r_body

    # Per-frame measured leg joints, expressed in the leg-local frame. The local
    # origin is the per-frame thorax-coxa when seen, else the body-fixed median.
    cols = [index.get(name, -1) for name in chain.point_names]
    world = np.stack(
        [pts3d[:, c] if c >= 0 else np.full((n_frames, 3), np.nan) for c in cols],
        axis=1,
    )  # (T, J, 3)
    root = world[:, 0]
    per_frame_origin = np.where(
        np.isfinite(root).all(axis=-1, keepdims=True), root, origin
    )
    local = to_local(world, per_frame_origin[:, None, :], r_body)  # (T, J, 3)

    angles = np.full((n_frames, n_dofs), np.nan)
    model_local = np.full((n_frames, n_joints, 3), np.nan)
    seed = np.clip((lo + hi) / 2.0, lo, hi)
    lo_x = lo + 1e-9
    hi_x = hi - 1e-9

    for t in range(n_frames):
        meas = local[t]
        mask = np.isfinite(meas).all(axis=-1)
        if int(mask.sum()) < _MIN_VALID_JOINTS:
            continue
        meas0 = np.where(mask[:, None], meas, 0.0)

        def residual(x, meas0=meas0, mask=mask):
            pred = np.asarray(fk(x, axes, seglens))
            return (mask[:, None] * (pred - meas0)).reshape(-1)

        def jacobian(x, mask=mask):
            j = np.asarray(jac(x, axes, seglens))  # (J, 3, D)
            return (mask[:, None, None] * j).reshape(-1, n_dofs)

        x0 = np.clip(seed, lo_x, hi_x)
        res = least_squares(
            residual,
            x0,
            jac=jacobian,
            bounds=(lo, hi),
            method="trf",
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
        )
        angles[t] = res.x
        model_local[t] = np.asarray(fk(res.x, axes, seglens))
        seed = res.x  # temporal seed for the next frame

    model_world = to_world(model_local, per_frame_origin[:, None, :], r_body)
    return angles, model_world


def solve_chain(
    pts3d: np.ndarray,
    index: dict[str, int],
    chain: Chain,
    body_sim: tuple[np.ndarray, float, np.ndarray],
    *,
    max_nfev: int = 100,
    loss: str = "linear",
    f_scale: float = 1.0,
    regularization: float = 0.01,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a baked articulation chain's joint angles (the head or abdomen) per frame.

    The chain is solved in the model frame: the measured markers (antenna tips /
    abdomen keypoints) are mapped through the inverse of the body similarity
    ``body_sim`` (model -> world), then a bounded least-squares fit
    (:func:`scipy.optimize.least_squares`, analytic JAX Jacobian) recovers the joint
    angles whose forward kinematics reach them. A small Tikhonov term pulls the
    angles toward the previous frame's solution, which both smooths the trajectory
    and pins redundant DOFs that the sparse markers do not constrain (e.g. the
    abdomen's interior hinges).

    Parameters
    ----------
    pts3d
        The 3D pose ``(T, P, 3)`` in world coordinates (NaN for missing points).
    index
        ``point_name -> column`` map into ``pts3d``.
    chain
        The baked articulation chain (joints + markers).
    body_sim
        ``(R, s, t)`` similarity mapping the neutral model onto the recording
        (from :func:`deeperfly.inverse_kinematics.articulation.body_similarity`).
    max_nfev, loss, f_scale
        Forwarded to :func:`scipy.optimize.least_squares`.
    regularization
        Weight of the toward-previous-frame angle prior (0 disables it).
    scale
        Data-estimated chain size relative to the model (see
        :func:`deeperfly.inverse_kinematics.articulation.estimate_chain_scale`). The
        neutral markers are grown about the chain's base anchor by this factor before
        the fit, so the angles reach the markers of a head/abdomen that differs in
        size from the fixed model geometry (and the overlay mesh, grown the same way,
        stays consistent). ``1.0`` keeps the model size.

    Returns
    -------
    angles : np.ndarray
        ``(T, D)`` fitted joint angles in radians (NaN for un-fittable frames).
    model_world : np.ndarray
        ``(T, M, 3)`` forward-kinematic marker positions in world coordinates.
    """
    rot, body_scale, trans = body_sim
    n_frames = pts3d.shape[0]
    n_dofs = len(chain.dof_names)
    n_markers = len(chain.marker_names)
    fk, jac = make_chain_kernels(chain.anchors, chain.axes, chain.marker_depth)
    # Grow the neutral markers about the chain base by the data-estimated size, the
    # same anchor the overlay mesh scales about (NmfMesh._node_transforms).
    base = np.asarray(chain.anchors[0], dtype=float)
    markers_neutral = base + float(scale) * (
        np.asarray(chain.marker_neutral, dtype=float) - base
    )
    lo, hi = chain.bounds

    cols = [index.get(name, -1) for name in chain.marker_names]
    world = np.stack(
        [pts3d[:, c] if c >= 0 else np.full((n_frames, 3), np.nan) for c in cols],
        axis=1,
    )  # (T, M, 3) measured markers in world
    # Map measured world markers into the model frame: m = R^T (w - t) / s.
    local = ((world - trans) @ rot) / max(body_scale, 1e-12)  # (T, M, 3)

    angles = np.full((n_frames, n_dofs), np.nan)
    model_local = np.full((n_frames, n_markers, 3), np.nan)
    sqrt_reg = float(np.sqrt(max(regularization, 0.0)))
    seed = np.clip(0.5 * (lo + hi), lo, hi)  # mid-range (0 for symmetric bounds)
    lo_x, hi_x = lo + 1e-9, hi - 1e-9
    eye = np.eye(n_dofs)

    for t in range(n_frames):
        meas = local[t]
        mask = np.isfinite(meas).all(axis=-1)
        if int(mask.sum()) < _MIN_VALID_JOINTS:
            continue
        meas0 = np.where(mask[:, None], meas, 0.0)
        prior = seed

        def residual(x, meas0=meas0, mask=mask, prior=prior):
            pred = np.asarray(fk(x, markers_neutral))
            r_mark = (mask[:, None] * (pred - meas0)).reshape(-1)
            return np.concatenate([r_mark, sqrt_reg * (x - prior)])

        def jacobian(x, mask=mask):
            j = np.asarray(jac(x, markers_neutral))  # (M, 3, D)
            j_mark = (mask[:, None, None] * j).reshape(-1, n_dofs)
            return np.vstack([j_mark, sqrt_reg * eye])

        res = least_squares(
            residual,
            np.clip(seed, lo_x, hi_x),
            jac=jacobian,
            bounds=(lo, hi),
            method="trf",
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
        )
        angles[t] = res.x
        model_local[t] = np.asarray(fk(res.x, markers_neutral))
        seed = res.x  # temporal seed + prior for the next frame

    model_world = body_scale * (model_local @ rot.T) + trans
    return angles, model_world


def solve_head(
    pts3d: np.ndarray,
    index: dict[str, int],
    head: HeadSpec,
    align: Alignment,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Antenna pitch/yaw per side by the vector method, plus the model tip points.

    Each antenna's angle is the elevation (pitch) and horizontal bearing (yaw) of
    the direction from the body-anterior head origin to the antenna tip, in the
    body-local frame (rest direction ``+x``, anterior). A no-op (empty outputs)
    when the head origin or the antenna points are unavailable.

    Returns
    -------
    angles : np.ndarray
        ``(T, D_head)`` antenna angles (radians), or shape ``(T, 0)``.
    angle_names : list of str
        The ``D_head`` angle names.
    model_world : np.ndarray
        ``(T, M_head, 3)`` antenna-tip world positions (passthrough of the data).
    point_names : list of str
        The ``M_head`` antenna point names (skeleton order).
    """
    n_frames = pts3d.shape[0]
    present = [a for a in head.antennae if a in index]
    if align.head_origin is None or not present:
        return np.zeros((n_frames, 0)), [], np.zeros((n_frames, 0, 3)), []

    r_body = align.r_body
    origin = align.head_origin
    angle_cols: list[np.ndarray] = []
    angle_names: list[str] = []
    tip_cols: list[np.ndarray] = []
    for name in present:
        tip = pts3d[:, index[name]]  # (T, 3)
        d_local = to_local(tip, origin, r_body)  # (T, 3)
        horiz = np.linalg.norm(d_local[:, :2], axis=-1)
        yaw = np.arctan2(d_local[:, 1], d_local[:, 0])
        pitch = np.arctan2(d_local[:, 2], horiz)
        angle_cols += [pitch, yaw]
        # The antenna tip direction maps onto flygym's antenna (pedicel) head joint.
        side = name.split("_")[0]  # "l" / "r"
        angle_names += [f"c_head-{side}_pedicel-pitch", f"c_head-{side}_pedicel-yaw"]
        tip_cols.append(tip)

    angles = np.stack(angle_cols, axis=1) if angle_cols else np.zeros((n_frames, 0))
    model_world = np.stack(tip_cols, axis=1) if tip_cols else np.zeros((n_frames, 0, 3))
    return angles, angle_names, model_world, present
