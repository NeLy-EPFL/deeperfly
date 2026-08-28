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
  matches the real fly's proportions and reprojects tightly -- optionally shared
  between each leg and its mirror image (:func:`symmetrize_seglens`), because a fly's
  left and right femurs are the same bone measured twice.

Per frame the solver transforms a leg's measured joints into the leg-local frame
(:func:`to_local`); the fitted model joints are mapped back to world
(:func:`to_world`) for reprojection and the overlay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..skeleton import Skeleton
from .template import KinematicTemplate

__all__ = [
    "Alignment",
    "body_alignment",
    "mirror_leg_pairs",
    "symmetrize_seglens",
    "to_local",
    "to_world",
]

log = logging.getLogger("deeperfly")

#: A measured length at or below this is "never observed" -- the same convention
#: :func:`~deeperfly.inverse_kinematics.bodyplan._model_seglens` reads it by, where such a
#: segment falls back to the model's own bone.
_EPS = 1e-9


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
    pts3d: np.ndarray,
    skeleton: Skeleton,
    template: KinematicTemplate,
    *,
    symmetric_segments: bool = False,
) -> Alignment:
    """Estimate the body frame + per-leg origins and segment lengths from the pose.

    Parameters
    ----------
    pts3d
        The reconstructed 3D pose, shape ``(T, P, 3)`` in world coordinates
        (NaN for un-triangulated points).
    skeleton
        The skeleton (resolves point names to columns of ``pts3d``, and declares which
        points mirror each other).
    template
        The kinematic template (its legs name the thorax-coxa / joint points).
    symmetric_segments
        Give each leg and its mirror image the same segment lengths
        (:func:`symmetrize_seglens`). Off by default, which measures the two sides
        independently.

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

    if symmetric_segments:
        seglens = symmetrize_seglens(seglens, template, skeleton)

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


def mirror_leg_pairs(
    template: KinematicTemplate, skeleton: Skeleton
) -> tuple[tuple[str, str], ...]:
    """The template's legs paired with their mirror images, per the skeleton.

    Two legs pair when *every* joint of one mirrors the joint at the same depth of the
    other, by the skeleton's own ``symmetries`` -- the same declared relation the
    training mirror augmentation reads (:mod:`deeperfly.skeleton`). Derived rather than matched on the ``l``/``r`` name
    prefix on purpose: the prefix is a convention of *this* template and this skeleton,
    while the symmetry pairs are the config's explicit statement of which point is which
    point's mirror image, and a skeleton that declares none is stating that its subject
    is not bilaterally symmetric.

    Returns each pair once, in template leg order, with the ``symmetries``-ordered
    partner second. A leg whose partner is not in the template (``legs = ["rf", "lf",
    "rm"]``) is absent, as is any leg in a skeleton that declares no pairs.
    """
    by_name = {leg.name: leg for leg in template.legs}
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for leg in template.legs:
        if leg.name in seen:
            continue
        partners = {
            skeleton.partner(point) if point in skeleton.point_names else None
            for point in leg.point_names
        }
        if None in partners:  # some joint of this leg has no mirror point
            continue
        names = {skeleton.point_names[i] for i in partners}  # type: ignore[index]
        match = next(
            (
                other.name
                for other in template.legs
                if other.name != leg.name and set(other.point_names) == names
            ),
            None,
        )
        if match is None:
            continue
        # Same depth as well as the same set: a chain paired point-for-point in the
        # wrong ORDER would share lengths between different bones.
        if any(
            skeleton.partner(a) != skeleton.point_names.index(b)
            for a, b in zip(leg.point_names, by_name[match].point_names)
        ):
            continue
        out.append((leg.name, match))
        seen |= {leg.name, match}
    return tuple(out)


def symmetrize_seglens(
    seglens: dict[str, np.ndarray],
    template: KinematicTemplate,
    skeleton: Skeleton,
) -> dict[str, np.ndarray]:
    """Give each leg and its mirror image one shared length per segment.

    A fly's left and right femurs are the same bone measured twice, so at most one of the
    two measured lengths can be anatomy. Averaging the two medians is what does not
    privilege a side (the same argument :func:`~deeperfly.postprocess.symmetrize_3d`
    makes for the body-fixed points, and the reason this is a *mean* rather than a pooled
    median over both sides' frames: pooling weights the side with more triangulated
    frames).

    A segment measured on only one side adopts that side's length rather than averaging
    a zero in -- which is strictly better than what happens without symmetry, where an
    unmeasured segment falls back to the model's own generic bone
    (:func:`~deeperfly.inverse_kinematics.bodyplan._model_seglens`). A segment measured
    on neither side is left at ``0.0`` for that fallback to pick up.

    This is a constraint on the *fly*, not on its pose: it fixes what the two legs are,
    and says nothing about what they are doing. Nothing here couples the two sides'
    joint angles, which is right -- a leg's left/right asymmetry at any instant IS the
    behavior.

    **It is a prior, and it costs.** Every point of a leg chain is tracked, so the chain
    is over-determined and the per-leg measured lengths already *are* the best fit to the
    keypoints; a shared length can only move off them. On the eight-view example
    recording, whose femurs measure 4-6% apart on all three pairs, sharing them raised
    the 3D residual 22% and the reprojection 0.14 px in 7 of 8 views -- and *raised* the
    left/right gap in each DOF's median angle from 4.7 to 6.4 degrees, which is the
    opposite of the hypothesis that a length error the solve cannot express as length
    comes out as angle. Hence off by default, and worth turning on for what it makes true
    of the model (one animal, comparable across sides) rather than for accuracy.

    Returns a new mapping; ``seglens`` is not modified.
    """
    pairs = mirror_leg_pairs(template, skeleton)
    out = {name: np.array(v, dtype=float) for name, v in seglens.items()}
    if not pairs:
        log.warning(
            "inverse_kinematics: symmetric_segments is on, but no leg of template %r "
            "pairs with a mirror image under skeleton %r's symmetries, so the segment "
            "lengths are unchanged",
            template.name,
            skeleton.name,
        )
        return out

    report: list[str] = []
    for left, right in pairs:
        la, lb = out.get(left), out.get(right)
        if la is None or lb is None:  # a leg with no observed coxa has no lengths
            continue
        gaps: list[float] = []
        for j in range(1, min(la.size, lb.size)):
            a, b = float(la[j]), float(lb[j])
            ok_a = np.isfinite(a) and a > _EPS
            ok_b = np.isfinite(b) and b > _EPS
            if ok_a and ok_b:
                shared = 0.5 * (a + b)
                gaps.append(abs(a - b) / shared)
            elif ok_a or ok_b:
                shared = a if ok_a else b
            else:
                continue  # never measured on either side: leave the model fallback
            la[j] = lb[j] = shared
        if gaps:
            report.append(f"{left}|{right} {max(gaps):.1%} worst")
    # The gap the sides HAD is the only check on the premise: two sides already
    # agreeing to within the noise had nothing to share, and a large gap says how much
    # of the fit's residual was the two triplets disagreeing about one bone.
    log.info(
        "inverse kinematics: segment lengths shared across %d mirror leg pair(s); "
        "the sides had differed by %s",
        len(pairs),
        ", ".join(report) if report else "(nothing measured on both sides)",
    )
    return out


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
