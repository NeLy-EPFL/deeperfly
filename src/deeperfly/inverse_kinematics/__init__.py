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
from dataclasses import dataclass

import numpy as np

from ..skeleton import Skeleton
from .align import Alignment, body_alignment
from .core import solve_head, solve_leg
from .template import KinematicTemplate

__all__ = [
    "IKResult",
    "KinematicTemplate",
    "Alignment",
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
        The ``D`` angle names (``Angle_<LEG>_<joint>_<dof>``), in column order.
    model_pts3d
        ``(T, P, 3)`` forward-kinematic joint positions in world coordinates,
        laid out in the skeleton's point order (leg joints + antenna tips filled,
        other points NaN) so they reproject with the skeleton's own bones.
    alignment
        The body frame + per-leg geometry used to register the pose.
    """

    angles: np.ndarray
    angle_names: list[str]
    model_pts3d: np.ndarray
    alignment: Alignment


def solve_inverse_kinematics(
    pts3d: np.ndarray,
    skeleton: Skeleton,
    template: KinematicTemplate,
    *,
    max_nfev: int = 100,
    loss: str = "linear",
    f_scale: float = 1.0,
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
        The kinematic template to fit.
    max_nfev, loss, f_scale
        Forwarded to the per-frame :func:`scipy.optimize.least_squares` solve.

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

    if template.head is not None:
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
    )
