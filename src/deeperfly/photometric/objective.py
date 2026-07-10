"""Cross-side rigid-pose parameterization and far-leg bone selection.

The diagnosed defect is a single weakly-constrained degree of freedom: the relative
pose between the two internally-consistent camera clusters (left ``{lf,lm,lh}`` and
right ``{rf,rm,rh}``), bridged only by the front camera. So the free variable is one
6-DOF rigid transform ``T = (rvec, tvec)`` applied to the *left* cluster (its cameras
**and** its triangulated 3D points move together, keeping intra-left reprojection
invariant); the right cluster and the front camera are the fixed gauge.

For a fixed (right/front) camera, a moved left point projects as ``project(T @ p)``; for
a left camera, :func:`compensate_left` gives the equivalent moved extrinsics so it still
images its own (moved) points identically while imaging fixed right points through
``T``. Both cross-side terms therefore depend on ``T``, and both intra-side terms are
invariant by construction.
"""

from __future__ import annotations

import cv2
import numpy as np
from jaxtyping import Float

# Per-leg-segment chamfer weights. Femur/tibia are long, straight and stick away from
# the body (unambiguous); the tarsus bends (the straight-bone model is poor) so it is
# down-weighted; the coxa sits in body clutter and is excluded.
SEGMENT_WEIGHTS = {"coxa": 0.0, "femur": 1.0, "tibia": 1.0, "tarsus": 0.3}


def rvec_to_rmat(rvec: Float[np.ndarray, "3"]) -> Float[np.ndarray, "3 3"]:
    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]


def rmat_to_rvec(rmat: Float[np.ndarray, "3 3"]) -> Float[np.ndarray, "3"]:
    return cv2.Rodrigues(np.asarray(rmat, float))[0].ravel()


def rigid(
    pts: Float[np.ndarray, "... 3"], delta: Float[np.ndarray, "6"]
) -> Float[np.ndarray, "... 3"]:
    """Apply ``T = (delta[:3] axis-angle, delta[3:] translation)`` to world points."""
    R = rvec_to_rmat(delta[:3])
    return pts @ R.T + delta[3:]


def compensate_left(
    rvec_c: Float[np.ndarray, "3"],
    tvec_c: Float[np.ndarray, "3"],
    delta: Float[np.ndarray, "6"],
) -> tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3"]]:
    """Left-camera extrinsics after applying relative-``T`` ``delta``.

    Chosen so the camera images its moved left points identically (intra-left
    invariant): ``R' = R_c R_T^T``, ``t' = t_c - R' delta_t``.
    """
    Rc = rvec_to_rmat(rvec_c)
    RT = rvec_to_rmat(delta[:3])
    Rp = Rc @ RT.T
    tp = np.asarray(tvec_c, float) - Rp @ delta[3:]
    return rmat_to_rvec(Rp), tp


def project_np(
    pts3d: Float[np.ndarray, "N 3"],
    rvec: Float[np.ndarray, "3"],
    tvec: Float[np.ndarray, "3"],
    kmat: Float[np.ndarray, "3 3"],
    dist: Float[np.ndarray, "K"],
) -> Float[np.ndarray, "N 2"]:
    """Project world points to raw-frame pixels via the OpenCV pinhole+distortion model."""
    d = None if (dist is None or len(dist) == 0) else np.asarray(dist, float)
    pts = np.ascontiguousarray(np.asarray(pts3d, float).reshape(-1, 1, 3))
    px, _ = cv2.projectPoints(
        pts,
        np.asarray(rvec, float).reshape(3, 1),
        np.asarray(tvec, float).reshape(3, 1),
        np.asarray(kmat, float),
        d,
    )
    return px.reshape(-1, 2)


def _segment_type(point_names, a: int, b: int) -> str:
    s = {point_names[a].split("_", 1)[1], point_names[b].split("_", 1)[1]}
    if s == {"thorax_coxa", "coxa_trochanter"}:
        return "coxa"
    if s == {"coxa_trochanter", "femur_tibia"}:
        return "femur"
    if s == {"femur_tibia", "tibia_tarsus"}:
        return "tibia"
    if s == {"tibia_tarsus", "claw"}:
        return "tarsus"
    return "other"


def left_point_indices(skeleton) -> np.ndarray:
    """Skeleton point indices belonging to left-side limbs (moved by ``T``)."""
    idx = [
        p
        for p in range(len(skeleton.point_names))
        if skeleton.limb_id[p] >= 0
        and skeleton.limb_names[skeleton.limb_id[p]].startswith("l")
    ]
    return np.asarray(idx, dtype=int)


def far_leg_bones(skeleton, cam_side: str, weights=SEGMENT_WEIGHTS):
    """Leg bones on the *far* side of a camera, as ``(a, b, weight)`` with weight>0.

    ``cam_side`` in ``{"left","right","front"}``. On the front camera only left legs
    carry a ``T`` dependence (moved points on a fixed camera), so they are the far set.
    """
    out = []
    for a, b in skeleton.bones:
        a, b = int(a), int(b)
        limb = skeleton.limb_names[skeleton.limb_id[a]]
        if not limb.endswith("_leg"):
            continue
        bone_side = "left" if limb.startswith("l") else "right"
        is_far = (
            (bone_side == "left") if cam_side == "front" else (bone_side != cam_side)
        )
        if not is_far:
            continue
        w = weights.get(_segment_type(skeleton.point_names, a, b), 0.0)
        if w > 0:
            out.append((a, b, float(w)))
    return out
