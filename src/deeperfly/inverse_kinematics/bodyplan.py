"""Build the QuickIK body plan the inverse-kinematics stage solves.

QuickIK is told *what* to fit by a body plan: a JSON kinematic tree in which every
joint is also a tracked keypoint, carrying its constant offset from its parent, its
ordered degrees of freedom (axis, neutral value, limits) and any zero-DOF
"pseudo-joints" needed where a keypoint sits mid-segment. :func:`build_body_plan`
assembles one for a recording out of three sources deeperfly already has:

- the leg spec (:mod:`deeperfly.inverse_kinematics.template`) -- DOF axes, joint limits
  and the flygym angle names, per leg and per side;
- the baked head/abdomen chains (:mod:`deeperfly.inverse_kinematics.articulation`) --
  neutral anchors, axes, limits and the markers (antenna tips, abdomen points) rigidly
  attached at a given chain depth;
- measurements from the data (:mod:`deeperfly.inverse_kinematics.align`) -- each leg's
  median coxa position and *measured* segment lengths, so the fitted model matches this
  fly's proportions and reprojects tightly rather than carrying generic model geometry.

Because the segment lengths are measured, the plan is built per recording (and held in
memory -- QuickIK parses it from a string) rather than shipped as a static asset.

**The plan lives in the model frame.** The legs are measured in the camera rig's world
frame at an arbitrary scale, while the baked chains are model-frame at model scale, and
the two body frames differ by a fixed rotation (~23 degrees of pitch: the coxa-centroid
frame :func:`~deeperfly.inverse_kinematics.align._body_axes` builds is not the model's
own frame) plus a per-recording scale. Mixing them under one root would misplace the
head and abdomen by a large fraction of their own extent, and their limits could not
absorb it. So everything is expressed in model coordinates: observations are mapped in
through the inverse of the body similarity
(:func:`~deeperfly.inverse_kinematics.articulation.body_similarity`) and fitted joints
are mapped back out. That also makes the solver's distance-based tolerances
rig-independent and keeps coordinates order-1, which matters because QuickIK is
``f32``.

The leg limits keep their exact meaning without rewriting a single axis: each leg's
thorax-coxa joint gets ``offset_quat`` = the *model's* coxa-derived body frame. A
joint's DOF axes live in its post-``offset_quat`` frame, so the template's
``yaw z / pitch y / roll x`` and its straight-down ``-z`` rest direction still mean what
they say, with one quaternion rotating the whole leg subtree into the model frame.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float

from .align import Alignment, _body_axes
from .articulation import Articulation, Chain
from .forward import REST_AXIS, PlanKinematics, rmat_to_quat
from .template import KinematicTemplate

__all__ = ["BodyPlan", "build_body_plan", "PLAN_VERSION"]

log = logging.getLogger("deeperfly")

#: Bumped when the generated plan's *structure* changes in a way that a stored plan
#: from an earlier deeperfly cannot be interpreted as. Carried in ``x-deeperfly``.
PLAN_VERSION = 1

#: The root joint. It sits at the model origin with an identity rotation: a fixed-base
#: plan then *is* the model frame (QuickIK leaves ``root_pos``/``root_rot`` untouched),
#: and a free-base plan lets the body move within it.
ROOT_NAME = "thorax"

_IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]
_EPS = 1e-9


@dataclass(frozen=True)
class BodyPlan:
    """A generated QuickIK body plan plus the bookkeeping its caller needs.

    Attributes
    ----------
    plan
        The body-plan mapping, JSON-serializable with no numpy left in it (see
        :meth:`to_json`).
    joint_names
        ``(N,)`` joint names in body-plan array order -- which is the order QuickIK
        expects observations in, and the order its DOF state is laid out in.
    joint_point
        ``(N,)`` the skeleton point each joint tracks, or ``None`` for a joint that is
        not a tracked keypoint (the root, and the head/abdomen hinges).
    joint_row
        ``(N,)`` each joint's column in the skeleton's point order, ``-1`` where it
        tracks nothing. The observation gather/scatter index.
    joint_branch, dof_branch
        ``(N,)`` / ``(D,)`` which limb each joint / DOF belongs to (a leg name, a chain
        name, or ``""`` for the root). Used to mask a limb that was too sparsely
        observed to fit.
    angle_names
        ``(D,)`` the flygym angle names in flat DOF order -- the ``angle_names``
        contract of ``results.h5``.
    neutral
        ``(D,)`` each DOF's neutral value, the solver's start pose and the target of
        its toward-neutral prior.
    body_sim
        ``(R, s, t)`` similarity mapping the *model* frame onto the recording's world
        frame; :meth:`to_model` / :meth:`to_world` apply it and its inverse.
    chain_scales
        ``chain name -> data-estimated size`` already baked into the plan's chain
        offsets.
    """

    plan: dict
    joint_names: tuple[str, ...]
    joint_point: tuple[str | None, ...]
    joint_row: np.ndarray
    joint_branch: tuple[str, ...]
    angle_names: tuple[str, ...]
    dof_branch: tuple[str, ...]
    neutral: np.ndarray
    body_sim: tuple[np.ndarray, float, np.ndarray]
    chain_scales: dict[str, float] = field(default_factory=dict)

    # -- views ---------------------------------------------------------------

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    @property
    def n_dofs(self) -> int:
        return len(self.angle_names)

    @property
    def fixed_base(self) -> bool:
        return bool(self.plan["fixed_base"])

    def kinematics(self) -> PlanKinematics:
        """Compile the plan for numpy forward kinematics."""
        return PlanKinematics.from_plan(self.plan)

    # -- frames --------------------------------------------------------------

    def to_model(
        self, world_pts: Float[np.ndarray, "*batch 3"]
    ) -> Float[np.ndarray, "*batch 3"]:
        """World points -> the model frame: ``R^T (p - t) / s``."""
        rot, scale, trans = self.body_sim
        return ((np.asarray(world_pts, dtype=float) - trans) @ rot) / max(scale, _EPS)

    def to_world(
        self, model_pts: Float[np.ndarray, "*batch 3"]
    ) -> Float[np.ndarray, "*batch 3"]:
        """Model points -> the world frame: ``s R p + t`` (inverse of :meth:`to_model`)."""
        rot, scale, trans = self.body_sim
        return scale * (np.asarray(model_pts, dtype=float) @ rot.T) + trans

    # -- serialization -------------------------------------------------------

    def to_json(self) -> str:
        """The plan as JSON -- what QuickIK parses and what ``results.h5`` stores.

        Deliberately without a ``default=`` fallback: a stray numpy value would
        otherwise be silently stringified (``json.dumps`` renders an array as
        ``"[1. 2.]"``), losing type and precision with no error. Here it raises.
        """
        return json.dumps(self.plan, separators=(",", ":"), sort_keys=False)

    @classmethod
    def from_json(cls, text: str, skeleton) -> "BodyPlan":
        """Rebuild a :class:`BodyPlan` from a stored plan and a skeleton.

        Lets the GUI re-solve on *exactly* the geometry the pipeline fitted, instead of
        re-deriving it (which drifts: the pipeline pins ``constant_points`` before
        measuring, and a live editor's pose is not pinned).
        """
        plan = json.loads(text)
        meta = plan.get(f"x-{_META_KEY}") or {}
        index = {name: i for i, name in enumerate(skeleton.point_names)}
        names, points, branches, rows = [], [], [], []
        angle_names, dof_branch, neutral = [], [], []
        for j in plan["joints"]:
            names.append(str(j["name"]))
            point = j.get("x-deeperfly-point")
            points.append(None if point is None else str(point))
            branch = str(j.get("x-deeperfly-branch", ""))
            branches.append(branch)
            rows.append(-1 if point is None else index.get(str(point), -1))
            for d in j.get("dofs") or ():
                angle_names.append(str(d["x-deeperfly-angle"]))
                dof_branch.append(branch)
                neutral.append(float(d["neutral"]))
        sim = meta.get("body_sim") or {}
        body_sim = (
            np.asarray(sim.get("rot", np.eye(3)), dtype=float).reshape(3, 3),
            float(sim.get("scale", 1.0)),
            np.asarray(sim.get("trans", np.zeros(3)), dtype=float),
        )
        return cls(
            plan=plan,
            joint_names=tuple(names),
            joint_point=tuple(points),
            joint_row=np.asarray(rows, dtype=np.int64),
            joint_branch=tuple(branches),
            angle_names=tuple(angle_names),
            dof_branch=tuple(dof_branch),
            neutral=np.asarray(neutral, dtype=float),
            body_sim=body_sim,
            chain_scales={
                str(k): float(v) for k, v in (meta.get("chain_scales") or {}).items()
            },
        )


_META_KEY = "deeperfly"


# -- construction ------------------------------------------------------------


def build_body_plan(
    template: KinematicTemplate,
    skeleton,
    alignment: Alignment,
    body_sim: tuple[np.ndarray, float, np.ndarray],
    *,
    articulation: Articulation | None = None,
    chain_scales: dict[str, float] | None = None,
    fixed_body: bool = True,
) -> BodyPlan:
    """Assemble the body plan for one recording.

    Parameters
    ----------
    template
        The leg spec (which legs, their DOF axes, limits and angle names).
    skeleton
        The skeleton, for resolving point names to observation columns.
    alignment
        The measured geometry: each leg's median coxa world position and measured
        segment lengths (:func:`~deeperfly.inverse_kinematics.align.body_alignment`).
    body_sim
        ``(R, s, t)`` mapping the model frame onto this recording's world frame
        (:func:`~deeperfly.inverse_kinematics.articulation.body_similarity`).
    articulation
        The baked head/abdomen chains to include; ``None`` builds a legs-only plan.
    chain_scales
        ``chain name -> size relative to the model``, baked into that chain's offsets
        so the fit and the mesh overlay agree (the overlay scales about the same base
        anchor). Missing entries mean model size.
    fixed_body
        Whether the root is fixed in the model frame. ``True`` suits a tethered fly
        (the body does not move, so the six body-fixed coxae pin it); ``False`` gives
        QuickIK a free 6-DOF root to fit per frame.

    Returns
    -------
    BodyPlan
        The plan plus the joint/DOF bookkeeping its caller needs.
    """
    scales = dict(chain_scales or {})
    joints: list[dict] = [
        {
            "name": ROOT_NAME,
            "parent": None,
            "offset_pos": [0.0, 0.0, 0.0],
            "offset_quat": list(_IDENTITY_QUAT),
            "dofs": [],
            "x-deeperfly-point": None,
            "x-deeperfly-branch": "",
        }
    ]
    for leg in template.legs:
        joints += _leg_joints(leg, skeleton, alignment, body_sim)
    for chain in articulation.chains if articulation is not None else ():
        joints += _chain_joints(chain, scales.get(chain.name, 1.0))

    plan = {
        "fixed_base": bool(fixed_body),
        f"x-{_META_KEY}": {
            "version": PLAN_VERSION,
            "template": template.name,
            "chain_scales": {k: float(v) for k, v in scales.items()},
            "body_sim": {
                "rot": [[float(v) for v in row] for row in np.asarray(body_sim[0])],
                "scale": float(body_sim[1]),
                "trans": [float(v) for v in np.asarray(body_sim[2])],
            },
        },
        "joints": joints,
    }

    index = {name: i for i, name in enumerate(skeleton.point_names)}
    points = tuple(j["x-deeperfly-point"] for j in joints)
    return BodyPlan(
        plan=plan,
        joint_names=tuple(str(j["name"]) for j in joints),
        joint_point=points,
        joint_row=np.asarray(
            [-1 if p is None else index.get(p, -1) for p in points], dtype=np.int64
        ),
        joint_branch=tuple(str(j["x-deeperfly-branch"]) for j in joints),
        angle_names=tuple(
            str(d["x-deeperfly-angle"]) for j in joints for d in j["dofs"]
        ),
        dof_branch=tuple(
            str(j["x-deeperfly-branch"]) for j in joints for _ in j["dofs"]
        ),
        neutral=np.asarray(
            [float(d["neutral"]) for j in joints for d in j["dofs"]], dtype=float
        ),
        body_sim=(
            np.asarray(body_sim[0], dtype=float),
            float(body_sim[1]),
            np.asarray(body_sim[2], dtype=float),
        ),
        chain_scales=scales,
    )


def _leg_joints(leg, skeleton, alignment: Alignment, body_sim) -> list[dict]:
    """One leg's chain: the thorax-coxa at its measured place, then measured segments.

    The thorax-coxa carries ``offset_quat`` = the model's own coxa-derived body frame,
    so the template's axes and its straight-down rest direction apply verbatim inside
    the leg while the whole subtree sits in the model frame.
    """
    rot, scale, trans = body_sim
    quat = [float(v) for v in rmat_to_quat(_model_body_axes())]
    origin = alignment.leg_origin.get(leg.name)
    if origin is None or not np.all(np.isfinite(origin)):
        # No coxa was ever seen: fall back to the model's own neutral coxa so the leg
        # still exists in the plan (its observations will all be missing, and the
        # solver's output for it is masked out afterwards).
        coxa_model = _model_neutral(skeleton, leg.joints[0].point)
    else:
        coxa_model = ((np.asarray(origin, dtype=float) - trans) @ rot) / max(
            scale, _EPS
        )

    seglens = _model_seglens(leg, skeleton, alignment, scale)
    out: list[dict] = []
    for j, joint in enumerate(leg.joints):
        parent = ROOT_NAME if j == 0 else leg.joints[j - 1].point
        offset = (
            [float(v) for v in coxa_model]
            if j == 0
            else [float(v) for v in REST_AXIS * seglens[j]]
        )
        out.append(
            {
                "name": joint.point,
                "parent": parent,
                "offset_pos": offset,
                "offset_quat": quat if j == 0 else list(_IDENTITY_QUAT),
                "dofs": [
                    {
                        "type": "hinge",
                        "axis": [float(a) for a in dof.axis],
                        "neutral": float(
                            np.clip(0.5 * (dof.lo + dof.hi), dof.lo, dof.hi)
                        ),
                        "limits": [float(dof.lo), float(dof.hi)],
                        "x-deeperfly-angle": f"{joint.joint}-{dof.name}",
                    }
                    for dof in joint.dofs
                ],
                "x-deeperfly-point": joint.point,
                "x-deeperfly-branch": leg.name,
            }
        )
    return out


def _chain_joints(chain: Chain, scale: float) -> list[dict]:
    """A baked chain's hinges followed by its markers as zero-DOF pseudo-joints.

    The chain's anchors are absolute neutral-model positions, so successive
    parent-relative offsets are their differences -- exact here because every
    ``offset_quat`` in the chain is identity and every neutral is the *chain's own*
    rest pose, which makes this tree algebraically the same transform as
    :func:`~deeperfly.inverse_kinematics.forward.chain_affine`'s rotations about those
    neutral anchors.

    ``scale`` grows the chain about its base anchor -- the same anchor and the same
    transform the mesh overlay scales that chain's nodes about, so the fit and the
    overlay stay consistent.
    """
    anchors = np.asarray(chain.anchors, dtype=float)
    lo, hi = chain.bounds
    base = anchors[0]
    f = float(scale)
    out: list[dict] = []
    for i, name in enumerate(chain.dof_names):
        # offset_pos[0] is the (unscaled) base anchor; scaling about the base leaves it
        # in place and multiplies every later inter-anchor step.
        offset = base if i == 0 else f * (anchors[i] - anchors[i - 1])
        out.append(
            {
                "name": name,
                "parent": ROOT_NAME if i == 0 else chain.dof_names[i - 1],
                "offset_pos": [float(v) for v in offset],
                "offset_quat": list(_IDENTITY_QUAT),
                "dofs": [
                    {
                        "type": "hinge",
                        "axis": [float(a) for a in chain.axes[i]],
                        "neutral": float(np.clip(0.5 * (lo[i] + hi[i]), lo[i], hi[i])),
                        "limits": [float(lo[i]), float(hi[i])],
                        "x-deeperfly-angle": name,
                    }
                ],
                "x-deeperfly-point": None,
                "x-deeperfly-branch": chain.name,
            }
        )
    for m, point in enumerate(chain.marker_names):
        depth = int(chain.marker_depth[m])
        neutral = np.asarray(chain.marker_neutral[m], dtype=float)
        if depth <= 0:  # rigidly on the body: no chain joint moves it
            parent = ROOT_NAME
            offset = base + f * (neutral - base)
        else:
            parent = chain.dof_names[depth - 1]
            offset = f * (neutral - anchors[depth - 1])
        out.append(
            {
                "name": point,
                "parent": parent,
                "offset_pos": [float(v) for v in offset],
                "offset_quat": list(_IDENTITY_QUAT),
                "dofs": [],
                "x-deeperfly-point": point,
                "x-deeperfly-branch": chain.name,
            }
        )
    return out


def _model_body_axes() -> np.ndarray:
    """The model's own coxa-derived body frame (columns x anterior, y left, z dorsal).

    :func:`~deeperfly.inverse_kinematics.align._body_axes` applied to the *model's*
    neutral coxae. The template's leg axes and limits are expressed in that frame (it
    is what a recording's ``r_body`` estimates), so a leg subtree rotated by this
    quaternion keeps them meaning exactly what they say -- while the model frame it
    sits in is pitched about 23 degrees away from it.
    """
    from .articulation import load_articulation

    art = load_articulation()
    origins = {
        point.split("_")[0]: np.asarray(neutral, dtype=float)
        for point, neutral in zip(art.coxa_points, art.coxa_neutral)
    }
    return _body_axes(origins)


def _model_neutral(skeleton, point: str) -> np.ndarray:
    """A skeleton point's neutral model position (from the packaged overlay asset)."""
    from .mesh import load_nmf_mesh

    index = {name: i for i, name in enumerate(skeleton.point_names)}
    if point not in index:
        return np.zeros(3)
    return np.asarray(load_nmf_mesh().kp_neutral[index[point]], dtype=float)


def _model_seglens(
    leg, skeleton, alignment: Alignment, body_scale: float
) -> np.ndarray:
    """``(J,)`` measured segment lengths in model units, with a model fallback.

    ``align`` measures bone lengths in the world frame at the rig's arbitrary scale;
    dividing by the body scale expresses them in model units. A segment that was never
    observed measures ``0.0`` there -- a zero-length link whose DOFs would have no
    moment arm -- so it falls back to the model's own length for that bone.
    """
    measured = np.asarray(alignment.seglens.get(leg.name, ()), dtype=float)
    points = leg.point_names
    out = np.zeros(len(points))
    fallbacks: list[str] = []
    for j in range(1, len(points)):
        value = (
            float(measured[j]) / max(float(body_scale), _EPS)
            if j < measured.size
            else 0.0
        )
        if not np.isfinite(value) or value <= _EPS:
            a = _model_neutral(skeleton, points[j - 1])
            b = _model_neutral(skeleton, points[j])
            value = float(np.linalg.norm(b - a))
            fallbacks.append(points[j])
        out[j] = value
    if fallbacks:
        # One line per leg, not per bone: a leg that was never triangulated at all
        # would otherwise emit four near-identical warnings.
        log.warning(
            "inverse_kinematics: leg %r has %d unmeasured segment(s) (%s); using the "
            "model's own bone lengths for them",
            leg.name,
            len(fallbacks),
            ", ".join(fallbacks),
        )
    return out
