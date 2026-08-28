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
the two body frames differ by a fixed rotation (~23 degrees of pitch: the frame the
coxa centroids induce is not the model's own frame) plus a per-recording scale. Mixing them under one root would misplace the
head and abdomen by a large fraction of their own extent, and their limits could not
absorb it. So everything is expressed in model coordinates: observations are mapped in
through the inverse of the body similarity
(:func:`~deeperfly.inverse_kinematics.articulation.body_similarity`) and fitted joints
are mapped back out. That also makes the solver's distance-based tolerances
rig-independent and keeps coordinates order-1, which matters because QuickIK is
``f32``.

The leg limits keep their exact meaning with no rotation at all: every leg body in the
MJCF carries ``quat="1 0 0 0"``, so a leg's hinge axes ARE the model frame's, and the plan
already lives in the model frame. Each leg subtree therefore gets an identity
``offset_quat`` and the template's axes apply verbatim.

That is a correction. The leg subtree used to be rotated by a coxa-derived body frame,
which is pitched about 23 degrees away from the model's own. Combined with two swapped
axes in the template it left the fit unable to reproduce the model's resting posture at
all. (That estimate is gone entirely: nothing else consumed it.) Both are fixed together, and
:mod:`deeperfly.inverse_kinematics.template` carries the measurement.

A DOF's ``neutral`` is the model's own **spring reference** for that joint -- the resting
angle its MJCF spring holds it at -- not the midpoint of its limits. QuickIK uses
``neutral`` both as the pose it starts frame 0 from and as what ``neutral_weight`` pulls
toward, so it is what picks between the two branches of the ``(yaw, roll)`` double cover.
A midpoint is not a rest pose and does not select a branch on purpose.
"""

from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Bool, Float

from .align import Alignment
from .articulation import Articulation, Chain
from .forward import REST_AXIS, PlanKinematics
from .template import KinematicTemplate

__all__ = ["BodyPlan", "build_body_plan", "PLAN_VERSION"]

log = logging.getLogger("deeperfly")

#: Bumped when the generated plan's *structure* changes in a way that a stored plan
#: from an earlier deeperfly cannot be interpreted as. Carried in ``x-deeperfly``.
#:
#: 2 -- a chain may be placed on a measured base marker rather than the model's anchor
#: (``chain_offsets``), and a chain may carry a zero-DOF marker at depth 0 that no DOF
#: moves. A version-1 plan still *reads* correctly (a missing ``chain_offsets`` is no
#: shift), but it was solved on the registered base, so its head angles are not
#: comparable with a version-2 fit's.
PLAN_VERSION = 2

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
    chain_offsets
        ``chain name -> (3,)`` how far the recording's chain base sits from the model's
        own, in model units, already baked into the plan's chain offsets. Measured from
        the chain's base marker (the ``neck`` for the head); absent for a chain that has
        no such landmark, which leaves it on the registered base as before. The mesh
        overlay applies the same translation, so the fit and the overlay agree.
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
    chain_offsets: dict[str, np.ndarray] = field(default_factory=dict)

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

    @property
    def joint_is_base(self) -> Bool[np.ndarray, "N"]:
        """``(N,)`` which joints are their chain's **base landmark**.

        A base landmark measures where its chain sits; it is not evidence about the
        chain's angles. The head's ``neck`` is the case that matters: it sits on the
        head's rotation center, and rotation about a point leaves that point exactly
        where it was, so no head DOF can move it however well it is triangulated.

        Read by :func:`deeperfly.inverse_kinematics.unfittable_branches`, which must not
        accept it as evidence the chain was observed -- otherwise declaring both
        antennae absent would leave the head "fitted" from the neck alone, and the
        solver's neutral-biased yaw/pitch/roll would be reported as a measurement.

        Taken from the plan's own joints so a plan rebuilt from JSON agrees with a
        freshly built one. Deliberately **not** derived as "no DOF can move this joint":
        that is equally true of each leg's thorax-coxa under a fixed base, and excluding
        those would change how an amputated leg is reported -- a separate question, and
        one this module answers per-branch rather than per-DOF.
        """
        return np.asarray(
            [bool(j.get(f"x-{_META_KEY}-base")) for j in self.plan["joints"]],
            dtype=bool,
        )

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
        re-deriving it (which drifts: the pipeline measures the plan from a pose whose
        static keypoints are already collapsed to one position by the ``[postprocess]``
        chain, and a live editor's pose is not).
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
            chain_offsets={
                str(k): np.asarray(v, dtype=float).reshape(3)
                for k, v in (meta.get("chain_offsets") or {}).items()
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
    chain_offsets: dict[str, np.ndarray] | None = None,
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
    chain_offsets
        ``chain name -> (3,)`` model-unit translation putting that chain's base where
        the recording's base marker was measured, rather than where the body
        registration extrapolated it. Missing entries leave the chain on the registered
        base.
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
    shifts = {
        str(k): np.asarray(v, dtype=float) for k, v in (chain_offsets or {}).items()
    }
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
        joints += _leg_joints(leg, skeleton, alignment, body_sim, template.rest_axis)
    for chain in articulation.chains if articulation is not None else ():
        joints += _chain_joints(
            chain, scales.get(chain.name, 1.0), shifts.get(chain.name)
        )

    plan = {
        "fixed_base": bool(fixed_body),
        f"x-{_META_KEY}": {
            "version": PLAN_VERSION,
            "template": template.name,
            "chain_scales": {k: float(v) for k, v in scales.items()},
            "chain_offsets": {
                k: [float(x) for x in np.asarray(v).reshape(3)]
                for k, v in shifts.items()
            },
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
        chain_offsets=shifts,
    )


def _leg_joints(
    leg, skeleton, alignment: Alignment, body_sim, rest_axis=REST_AXIS
) -> list[dict]:
    """One leg's chain: the thorax-coxa at its measured place, then measured segments.

    A joint's ``offset_quat`` is the model body's own orientation, and a segment runs
    along the model's own ``rest_axis``. Both are identity/``-z`` for NeuroMechFly,
    whose leg bodies are axis-aligned with the model frame, which is what lets its
    template's axes mean what they say (see the module docstring) -- and both are read
    off the template rather than assumed, because flybody's are neither.
    """
    rot, scale, trans = body_sim
    rest = _leg_spring_reference()
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
            else [float(v) for v in np.asarray(rest_axis, dtype=float) * seglens[j]]
        )
        out.append(
            {
                "name": joint.point,
                "parent": parent,
                "offset_pos": offset,
                "offset_quat": [float(v) for v in joint.quat],
                "dofs": [
                    {
                        "type": "hinge",
                        "axis": [float(a) for a in dof.axis],
                        "neutral": float(
                            np.clip(rest.get(dof.angle, 0.0), dof.lo, dof.hi)
                        ),
                        "limits": [float(dof.lo), float(dof.hi)],
                        "x-deeperfly-angle": dof.angle,
                    }
                    for dof in joint.dofs
                ],
                "x-deeperfly-point": joint.point,
                "x-deeperfly-branch": leg.name,
            }
        )
    return out


def _chain_joints(
    chain: Chain, scale: float, shift: np.ndarray | None = None
) -> list[dict]:
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

    ``shift`` then translates the grown chain rigidly, putting its base where the
    recording's own base marker was measured instead of where the coxa registration
    extrapolated it (:attr:`~deeperfly.inverse_kinematics.articulation.Chain.base_point`).
    This is the chain's version of what :func:`_leg_joints` already does with each leg's
    measured median thorax-coxa, and it matters for the same reason: a chain root left
    at the model's own anchor inherits every error in the registration that placed it.
    Only the two *absolute* offsets move -- the root's, and those of markers parented
    straight to the root -- because every other offset in the chain is a difference
    between two anchors, which a rigid translation leaves alone. The mesh overlay
    applies the same translation to its node affines, where a rigid shift of a chain's
    anchors and its attached points is exactly a post-translation of ``(A, b)``.
    """
    anchors = np.asarray(chain.anchors, dtype=float)
    lo, hi = chain.bounds
    base = anchors[0]
    delta = np.zeros(3) if shift is None else np.asarray(shift, dtype=float)
    origin = base + delta
    f = float(scale)
    out: list[dict] = []
    for i, name in enumerate(chain.dof_names):
        # offset_pos[0] is the (unscaled) base anchor, shifted onto the measured base;
        # scaling about the base leaves it in place and multiplies every later
        # inter-anchor step.
        offset = origin if i == 0 else f * (anchors[i] - anchors[i - 1])
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
            offset = origin + f * (neutral - base)
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
                **({f"x-{_META_KEY}-base": True} if point == chain.base_point else {}),
            }
        )
    return out


@functools.cache
def _leg_spring_reference() -> dict[str, float]:
    """``angle name -> resting angle in radians``, the model's own spring references.

    Every leg hinge in the MJCF carries a ``springref``: the angle its passive spring
    holds the joint at, which is the model's resting posture and not the middle of its
    travel. :mod:`deeperfly.inverse_kinematics.articulation` bakes them, so this needs no
    MuJoCo at runtime.

    They are the plan's ``neutral``, which QuickIK uses for two things: the pose it starts
    the first frame from, and the target ``neutral_weight`` pulls every DOF toward. Both
    matter more than they look. The leg chain's ``(yaw, roll)`` pair has a double cover --
    ``(theta, r)`` and ``(-theta, r +- 180)`` place the leg identically -- and nothing in
    the keypoints distinguishes them, so whichever branch the solve starts nearest is the
    one the whole recording is reported in. Started from the midpoint of the limits, the
    mid and hind legs went to the mirror branch, where the required yaw is a further 90
    degrees out and the limits then clip it.

    Read from the packaged asset rather than an argument because a legs-only plan (no
    ``articulation=``) needs them just as much, and a leg's rest pose is a property of the
    model, not of what the caller chose to fit.
    """
    from .articulation import load_articulation

    return {k: float(v) for k, v in load_articulation().leg_rest.items()}


def _model_neutral(skeleton, point: str) -> np.ndarray:
    """A skeleton point's neutral model position (from the packaged overlay asset)."""
    from .mesh import load_model_mesh

    index = {name: i for i, name in enumerate(skeleton.point_names)}
    if point not in index:
        return np.zeros(3)
    return np.asarray(load_model_mesh().kp_neutral[index[point]], dtype=float)


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
