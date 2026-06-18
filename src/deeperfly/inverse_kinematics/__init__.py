"""Inverse kinematics: fit a NeuroMechFly-style model's joint angles to the 3D pose.

The ``inverse_kinematics`` pipeline stage runs after triangulation and recovers the
articulated joint angles whose forward kinematics best match the reconstructed 3D
keypoints. It is a self-contained reimplementation of NeLy-EPFL's
sequential-inverse-kinematics (no ikpy/flygym dependency): a kinematic
:class:`~deeperfly.inverse_kinematics.template.KinematicTemplate` describes the leg
chains and joint limits, :func:`~deeperfly.inverse_kinematics.align.body_alignment`
registers the recording to the template frame, and a bounded JAX/scipy least-squares
solve (:mod:`deeperfly.inverse_kinematics.core`) fits each leg's angles per frame.

The result carries the joint-angle trajectories *and* the fitted model's joint
positions in world coordinates -- so the model reprojects onto the raw 2D images
as an overlay (the GUI and the ``skeleton_nmf`` visualization op).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..skeleton import Skeleton
from .align import Alignment, body_alignment
from .articulation import Articulation, body_similarity, estimate_chain_scale
from .core import solve_chain, solve_head, solve_leg
from .mesh import NmfMesh, load_nmf_mesh
from .template import KinematicTemplate

__all__ = [
    "IKResult",
    "KinematicTemplate",
    "Articulation",
    "Alignment",
    "NmfMesh",
    "load_nmf_mesh",
    "solve_inverse_kinematics",
]

log = logging.getLogger("deeperfly")


@dataclass
class IKResult:
    """The fitted joint angles and the model pose they imply.

    Attributes
    ----------
    angles
        ``(T, D)`` joint angles in radians (NaN for un-fittable frames).
    angle_names
        The ``D`` angle names -- the flygym joint names ``<parent>-<child>-<dof>``
        (e.g. ``c_thorax-rf_coxa-roll``), in column order.
    model_pts3d
        ``(T, P, 3)`` forward-kinematic joint positions in world coordinates,
        laid out in the skeleton's point order (leg joints + antenna tips filled,
        other points NaN) so they reproject with the skeleton's own bones.
    alignment
        The body frame + per-leg geometry used to register the pose.
    chain_scales
        ``chain name -> data-estimated size`` (head / abdomen) relative to the model,
        used to grow those mesh parts in the overlay (the legs always skin to the
        measured keypoints). Empty when no articulation was fit.
    body_scale
        The overlay body scale, estimated **once** for the recording from the median
        thorax-coxa spread (the fly is rigid, so its size is constant). The mesh
        overlay then places the rigid body + head + abdomen per frame at this fixed
        size, varying only rotation + translation -- so the body no longer "breathes"
        with per-frame coxa noise. ``1.0`` when the coxae could not be registered.
    """

    angles: np.ndarray
    angle_names: list[str]
    model_pts3d: np.ndarray
    alignment: Alignment
    chain_scales: dict[str, float] = field(default_factory=dict)
    body_scale: float = 1.0


def solve_inverse_kinematics(
    pts3d: np.ndarray,
    skeleton: Skeleton,
    template: KinematicTemplate,
    *,
    articulation: Articulation | None = None,
    max_nfev: int = 100,
    loss: str = "linear",
    f_scale: float = 1.0,
    regularization: float = 0.01,
) -> IKResult:
    """Fit the template's joint angles to a 3D pose sequence.

    Parameters
    ----------
    pts3d
        The reconstructed 3D pose ``(T, P, 3)`` in world coordinates (NaN for
        un-triangulated points), in the ``skeleton``'s point order.
    skeleton
        The skeleton (resolves the template's point names to columns of ``pts3d``).
    template
        The kinematic template (leg chains) to fit.
    articulation
        The baked head/abdomen chains to additionally fit (``None`` skips them, and
        the legacy antenna vector method runs if the template names antennae).
    max_nfev, loss, f_scale
        Forwarded to the per-frame :func:`scipy.optimize.least_squares` solve.
    regularization
        Toward-previous-frame angle prior for the head/abdomen chains.

    Returns
    -------
    IKResult
        The joint angles, model joint positions, and the alignment.
    """
    pts3d = np.asarray(pts3d, dtype=float)
    n_frames, n_points = pts3d.shape[0], pts3d.shape[1]
    index = {name: i for i, name in enumerate(skeleton.point_names)}

    align = body_alignment(pts3d, skeleton, template)

    angle_cols: list[np.ndarray] = []
    angle_names: list[str] = []
    chain_scales: dict[str, float] = {}
    model = np.full((n_frames, n_points, 3), np.nan)

    for leg in template.legs:
        if leg.name not in align.leg_origin:
            log.warning(
                "inverse_kinematics: skipping leg %r (no thorax-coxa observed)",
                leg.name,
            )
            angle_cols.append(np.full((n_frames, sum(leg.dof_counts)), np.nan))
            angle_names += leg.dof_names
            continue
        leg_angles, leg_world = solve_leg(
            pts3d, index, leg, align, max_nfev=max_nfev, loss=loss, f_scale=f_scale
        )
        angle_cols.append(leg_angles)
        angle_names += leg.dof_names
        for j, name in enumerate(leg.point_names):
            if name in index:
                model[:, index[name]] = leg_world[:, j]

    if articulation is not None and articulation.chains:
        chain_scales = _solve_articulation(
            pts3d,
            index,
            articulation,
            angle_cols,
            angle_names,
            model,
            max_nfev=max_nfev,
            loss=loss,
            f_scale=f_scale,
            regularization=regularization,
        )
    elif template.head is not None:
        head_angles, head_names, head_world, head_points = solve_head(
            pts3d, index, template.head, align
        )
        if head_angles.shape[1]:
            angle_cols.append(head_angles)
            angle_names += head_names
            for m, name in enumerate(head_points):
                model[:, index[name]] = head_world[:, m]

    angles = (
        np.concatenate(angle_cols, axis=1) if angle_cols else np.zeros((n_frames, 0))
    )
    return IKResult(
        angles=angles,
        angle_names=angle_names,
        model_pts3d=model,
        alignment=align,
        chain_scales=chain_scales,
        body_scale=_estimate_body_scale(pts3d),
    )


def _estimate_body_scale(pts3d: np.ndarray) -> float:
    """The overlay body scale, fit once from the median thorax-coxa positions.

    The fly is a rigid body, so its overall size is constant over a recording. We
    register the overlay mesh's neutral coxae onto the *median* measured coxae and
    keep that one similarity scale; the per-frame mesh placement then varies only
    rotation + translation, instead of re-fitting scale each frame (which made the
    body breathe with coxa noise). ``1.0`` when too few coxae are seen.
    """
    mesh = load_nmf_mesh()
    with np.errstate(all="ignore"):
        median = np.nanmedian(pts3d, axis=0)  # (P, 3)
    return float(mesh._body_transform(median)[1])


def _solve_articulation(
    pts3d, index, articulation, angle_cols, angle_names, model, **kw
) -> dict[str, float]:
    """Fit the head/abdomen chains and fill their angles + reprojected markers.

    Registers the body once via the six thorax-coxa keypoints; a chain whose body
    cannot be placed (too few coxae) or whose markers are missing yields NaN angles.
    Each chain's size relative to the model is estimated from its markers (see
    :func:`~deeperfly.inverse_kinematics.articulation.estimate_chain_scale`), applied
    to the fit, and returned (so the overlay grows that part to match).
    """
    n_frames = pts3d.shape[0]
    coxa_cols = [index.get(p, -1) for p in articulation.coxa_points]
    coxa_world = np.stack(
        [pts3d[:, c] if c >= 0 else np.full((n_frames, 3), np.nan) for c in coxa_cols],
        axis=1,
    )
    with np.errstate(all="ignore"):
        measured_coxae = np.nanmedian(coxa_world, axis=0)  # (6, 3)
    sim = body_similarity(articulation.coxa_neutral, measured_coxae)

    scales: dict[str, float] = {}
    for chain in articulation.chains:
        if sim is None:
            log.warning(
                "inverse_kinematics: skipping %r chain (could not place the body)",
                chain.name,
            )
            angle_cols.append(np.full((n_frames, len(chain.dof_names)), np.nan))
            angle_names += list(chain.dof_names)
            continue
        scale = _chain_scale(pts3d, index, chain, sim)
        scales[chain.name] = scale
        ch_angles, ch_world = solve_chain(pts3d, index, chain, sim, scale=scale, **kw)
        angle_cols.append(ch_angles)
        angle_names += list(chain.dof_names)
        for m, name in enumerate(chain.marker_names):
            if name in index:
                model[:, index[name]] = ch_world[:, m]
    if scales:
        log.info(
            "inverse_kinematics: estimated chain scale %s",
            {k: round(v, 3) for k, v in scales.items()},
        )
    return scales


def _chain_scale(pts3d, index, chain, body_sim) -> float:
    """The chain's data-estimated size: measured markers mapped to the model frame."""
    rot, body_scale, trans = body_sim
    n_frames = pts3d.shape[0]
    cols = [index.get(name, -1) for name in chain.marker_names]
    world = np.stack(
        [pts3d[:, c] if c >= 0 else np.full((n_frames, 3), np.nan) for c in cols],
        axis=1,
    )
    local = ((world - trans) @ rot) / max(body_scale, 1e-12)
    return estimate_chain_scale(local, chain)
