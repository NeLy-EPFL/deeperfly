"""Register the reconstructed 3D pose to the kinematic template's frame.

The triangulated pose lives in the camera rig's world frame at an arbitrary scale
and orientation; the template's leg chains live in a body-local frame (anterior
``x``, left ``y``, dorsal ``z``) at the model's own scale. :func:`body_alignment`
estimates the rigid body frame and the per-leg geometry that bridge the two:

- a body rotation ``r_body`` (its columns are the body axes in world coordinates),
  derived from the six body-fixed thorax-coxa joints (a tethered fly's body does
  not move, so these are constant up to noise);
- each leg's coxa origin (the median thorax-coxa world position);
- each leg's *measured* segment lengths (median bone lengths), so the fitted chain
  matches the real fly's proportions and reprojects tightly.

Per frame the solver transforms a leg's measured joints into the leg-local frame
(:func:`to_local`); the fitted model joints are mapped back to world
(:func:`to_world`) for reprojection and the overlay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..skeleton import Skeleton
from .template import KinematicTemplate

__all__ = ["Alignment", "body_alignment", "to_local", "to_world"]


@dataclass(frozen=True)
class Alignment:
    """The body frame and per-leg geometry registering a recording to the template."""

    r_body: np.ndarray  # (3, 3) world-frame columns of the body axes (x, y, z)
    leg_origin: dict[str, np.ndarray]  # leg name -> (3,) coxa world position
    seglens: dict[str, np.ndarray]  # leg name -> (J,) segment length into each joint
    head_origin: np.ndarray | None  # (3,) body-anterior reference, or None

    def to_json(self) -> dict:
        return {
            "r_body": self.r_body.tolist(),
            "leg_origin": {k: v.tolist() for k, v in self.leg_origin.items()},
            "seglens": {k: v.tolist() for k, v in self.seglens.items()},
            "head_origin": None
            if self.head_origin is None
            else self.head_origin.tolist(),
        }


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _nanmedian_point(pts: np.ndarray) -> np.ndarray:
    """Median over the leading (time) axis of ``(T, 3)`` points, ignoring NaN rows."""
    with np.errstate(all="ignore"):
        out = np.nanmedian(pts, axis=0)
    return out


def body_alignment(
    pts3d: np.ndarray, skeleton: Skeleton, template: KinematicTemplate
) -> Alignment:
    """Estimate the body frame + per-leg origins and segment lengths from the pose.

    Parameters
    ----------
    pts3d
        The reconstructed 3D pose, shape ``(T, P, 3)`` in world coordinates
        (NaN for un-triangulated points).
    skeleton
        The skeleton (resolves point names to columns of ``pts3d``).
    template
        The kinematic template (its legs name the thorax-coxa / joint points).

    Returns
    -------
    Alignment
        The body rotation, per-leg coxa origins and measured segment lengths.
    """
    index = {name: i for i, name in enumerate(skeleton.point_names)}

    def coxa(leg) -> np.ndarray | None:
        name = leg.joints[0].point
        if name not in index:
            return None
        return _nanmedian_point(pts3d[:, index[name]])

    origins = {leg.name: coxa(leg) for leg in template.legs}
    r_body = _body_axes(origins)

    leg_origin: dict[str, np.ndarray] = {}
    seglens: dict[str, np.ndarray] = {}
    for leg in template.legs:
        if origins[leg.name] is None or not np.all(np.isfinite(origins[leg.name])):
            continue
        leg_origin[leg.name] = origins[leg.name]
        seglens[leg.name] = _measured_seglens(pts3d, index, leg)

    head_origin = _head_origin(origins)
    return Alignment(
        r_body=r_body,
        leg_origin=leg_origin,
        seglens=seglens,
        head_origin=head_origin,
    )


def _group_centroid(origins: dict, names: list[str]) -> np.ndarray | None:
    pts = [origins[n] for n in names if origins.get(n) is not None]
    pts = [p for p in pts if np.all(np.isfinite(p))]
    return np.mean(pts, axis=0) if pts else None


def _body_axes(origins: dict) -> np.ndarray:
    """Orthonormal body axes (columns x=anterior, y=left, z=dorsal) from the coxae.

    Falls back to the world axes for any direction the available coxae cannot
    determine, so a partial leg set still yields a usable (if cruder) frame.
    """
    right = _group_centroid(origins, ["rf", "rm", "rh"])
    left = _group_centroid(origins, ["lf", "lm", "lh"])
    front = _group_centroid(origins, ["lf", "rf"])
    hind = _group_centroid(origins, ["lh", "rh"])

    y = (
        _normalize(left - right)
        if left is not None and right is not None
        else np.array([0.0, 1.0, 0.0])
    )
    x = (
        _normalize(front - hind)
        if front is not None and hind is not None
        else np.array([1.0, 0.0, 0.0])
    )
    z = _normalize(np.cross(x, y))
    if np.linalg.norm(z) == 0:  # x and y parallel: pick a fallback dorsal axis
        z = np.array([0.0, 0.0, 1.0])
    # Re-orthogonalize so the frame is exactly orthonormal (y from the cross is
    # the most reliable; rebuild x to be perpendicular to both).
    x = _normalize(np.cross(y, z))
    y = _normalize(np.cross(z, x))
    return np.stack([x, y, z], axis=1)  # columns are the body axes in world


def _measured_seglens(pts3d: np.ndarray, index: dict, leg) -> np.ndarray:
    """Median measured bone length into each joint of a leg (0 for the root)."""
    lengths = [0.0]
    pts = leg.point_names
    for j in range(1, len(pts)):
        if pts[j - 1] in index and pts[j] in index:
            a = pts3d[:, index[pts[j - 1]]]
            b = pts3d[:, index[pts[j]]]
            seg = np.linalg.norm(b - a, axis=-1)
            seg = seg[np.isfinite(seg)]  # a never-observed segment -> length 0
            lengths.append(float(np.median(seg)) if seg.size else 0.0)
        else:
            lengths.append(0.0)
    return np.asarray(lengths, dtype=float)


def _head_origin(origins: dict) -> np.ndarray | None:
    """A body-anterior reference for the antenna vector method (front coxae centroid)."""
    front = _group_centroid(origins, ["lf", "rf"])
    return front


def to_local(
    world_pts: np.ndarray, origin: np.ndarray, r_body: np.ndarray
) -> np.ndarray:
    """World points -> the leg-local frame: ``r_body.T @ (p - origin)``.

    ``world_pts`` is ``(..., 3)``; ``origin`` is ``(3,)``; ``r_body`` is the body
    rotation (columns are the body axes in world).
    """
    return (world_pts - origin) @ r_body  # (p - o) @ R == R.T @ (p - o) per row


def to_world(
    local_pts: np.ndarray, origin: np.ndarray, r_body: np.ndarray
) -> np.ndarray:
    """Leg-local points -> world: ``r_body @ p_local + origin`` (inverse of :func:`to_local`)."""
    return local_pts @ r_body.T + origin
