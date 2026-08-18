"""Inverse kinematics: fit a NeuroMechFly-style model's joint angles to the 3D pose.

The ``inverse_kinematics`` pipeline stage runs after triangulation and recovers the
articulated pose behind the reconstructed keypoints: the joint angles whose forward
kinematics reach them. A 3D point cloud says *where* each keypoint is; the angles say
what the animal did.

The solve is `QuickIK <https://nely-epfl.github.io/quickik/>`_'s -- a Rust whole-body IK
library, installed as the optional ``deeperfly[ik]`` extra (see
:mod:`deeperfly.inverse_kinematics._quickik`). deeperfly's part is everything around it:

- :mod:`~deeperfly.inverse_kinematics.template` and
  :mod:`~deeperfly.inverse_kinematics.articulation` hold the model -- the leg chains'
  DOF axes and limits, and the baked head/abdomen geometry;
- :mod:`~deeperfly.inverse_kinematics.align` measures this recording -- each leg's
  median coxa and its *measured* segment lengths;
- :mod:`~deeperfly.inverse_kinematics.bodyplan` turns those into the body plan QuickIK
  solves, in the model frame;
- :mod:`~deeperfly.inverse_kinematics.forward` evaluates the fit back into joint
  positions (QuickIK's Python bindings expose angles and a root pose, not forward
  kinematics).

One solve fits the whole body -- six legs, head and abdomen -- against every tracked
keypoint at once, warm-started frame to frame, rather than fitting each limb on its own.

The result carries the joint-angle trajectories *and* the fitted model's joint positions
in world coordinates, so the model reprojects onto the raw 2D images as an overlay (the
GUI and the ``skeleton_nmf`` / ``mesh_nmf`` visualization ops).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Bool, Float

from ..config import InverseKinematicsParams
from ..skeleton import Skeleton
from ._quickik import MissingQuickIK, require_quickik
from .align import Alignment, body_alignment
from .articulation import (
    Articulation,
    body_similarity,
    calibrate_chain,
    estimate_chain_scale,
)
from .bodyplan import BodyPlan, build_body_plan
from .mesh import NmfMesh, load_nmf_mesh
from .template import KinematicTemplate

__all__ = [
    "IKResult",
    "KinematicTemplate",
    "Articulation",
    "Alignment",
    "BodyPlan",
    "observations",
    "unfittable_branches",
    "MissingQuickIK",
    "NmfMesh",
    "load_nmf_mesh",
    "solve_inverse_kinematics",
]

log = logging.getLogger("deeperfly")

#: A limb must have at least this many observed keypoints in a frame for its angles to
#: mean anything. QuickIK returns *an* angle for every DOF regardless (pulled toward
#: neutral), so without this the overlay would draw a confidently invented limb wherever
#: the data went missing.
_MIN_VALID_JOINTS = 2

#: The configured defaults, so this function's signature cannot drift from
#: ``[inverse_kinematics]``. Defaults are written exactly once, in
#: :class:`~deeperfly.config.InverseKinematicsParams`; a direct library call and a
#: pipeline run must not solve the same pose differently.
_D = InverseKinematicsParams()


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
        laid out in the skeleton's point order (leg joints, antenna tips and abdomen
        markers filled, other points NaN) so they reproject with the skeleton's own
        bones.
    alignment
        The body frame + per-leg geometry measured from the recording.
    body_plan
        The plan that was solved. Stored in ``results.h5`` so the editor's live re-fit
        runs on exactly this geometry rather than re-deriving it.
    chain_scales
        ``chain name -> data-estimated size`` (head / abdomen) relative to the model.
        Baked into the plan's chain offsets, and applied to the same chains by the mesh
        overlay, so the fit and the overlay agree. Empty when no chains were fit.
    chain_offsets
        ``chain name -> (3,)`` model-unit shift putting each chain's base where the
        recording's own base landmark was measured (the head's ``neck``) instead of
        where the coxa registration extrapolated it. Baked into the plan and applied to
        the mesh overlay for the same reason ``chain_scales`` is. Empty for a chain with
        no base landmark, which leaves it on the registered base as before.
    body_scale
        The recording's body size relative to the model, from the one coxa registration
        that also places the plan. The fly is rigid, so this is constant over the
        recording: the mesh overlay then varies only rotation and translation per frame
        instead of re-fitting scale (which made the body "breathe" with coxa noise).
    """

    angles: np.ndarray
    angle_names: list[str]
    model_pts3d: np.ndarray
    alignment: Alignment
    body_plan: BodyPlan | None = None
    chain_scales: dict[str, float] = field(default_factory=dict)
    chain_offsets: dict[str, np.ndarray] = field(default_factory=dict)
    body_scale: float = 1.0


def solve_inverse_kinematics(
    pts3d: Float[np.ndarray, "T P 3"],
    skeleton: Skeleton,
    template: KinematicTemplate,
    *,
    articulation: Articulation | None = None,
    weights: Float[np.ndarray, "T P"] | None = None,
    n_iterations: int = _D.n_iterations,
    neutral_weight: float = _D.neutral_weight,
    damping: float = _D.damping,
    position_tolerance: float = _D.position_tolerance,
    angle_tolerance: float = _D.angle_tolerance,
    fixed_body: bool = _D.fixed_body,
    parallel: bool = _D.parallel,
    segment_len: int = _D.segment_len,
    overlap_len: int = _D.overlap_len,
    absent_points: Bool[np.ndarray, "T P"] | None = None,
) -> IKResult:
    """Fit the model's joint angles to a 3D pose sequence.

    Parameters
    ----------
    pts3d
        The reconstructed 3D pose ``(T, P, 3)`` in world coordinates (NaN for
        un-triangulated points), in the ``skeleton``'s point order.
    skeleton
        The skeleton (resolves point names to columns of ``pts3d``).
    template
        The kinematic template (which legs to fit, their DOF axes and limits).
    articulation
        The baked head/abdomen chains to additionally fit; ``None`` fits legs only.
    weights
        Optional ``(T, P)`` per-observation weights (e.g. detector confidence).
        ``None`` weighs every observed point equally. A non-finite or non-positive
        weight marks the point unobserved, as a NaN position does.
    n_iterations, neutral_weight, damping, position_tolerance, angle_tolerance
        QuickIK's solver configuration. The plan is solved in model units, so the two
        tolerances are rig-independent.
    fixed_body
        Fix the body in the model frame (a tethered fly) instead of fitting a 6-DOF
        root per frame.
    parallel, segment_len, overlap_len
        Solve in overlapping segments across worker threads. Off by default: segments
        restart from the neutral pose, so the angle traces can step at a seam.
    absent_points
        ``(T, P)`` keypoints that are **not on this animal** (an amputated leg, an ablated
        antenna), per frame; a ``(P,)`` whole-recording declaration is accepted too. Used
        only to judge how many observations a branch could possibly deliver, so a stump is
        still fitted from what remains instead of being written off as an unobserved limb
        -- see :func:`unfittable_branches`. The **body plan** is built once per recording
        and so cannot vary in time: a leg lost part-way through stays in the plan, which is
        correct (it existed, and its angles up to the loss are the measurement).

    Returns
    -------
    IKResult
        The joint angles, the fitted model joints, the measurements and the plan.

    Raises
    ------
    MissingQuickIK
        If the optional ``deeperfly[ik]`` extra is not installed.
    """
    quickik = require_quickik()
    pts3d = np.asarray(pts3d, dtype=float)
    n_frames, n_points = pts3d.shape[0], pts3d.shape[1]

    alignment = body_alignment(pts3d, skeleton, template)
    plan = _plan_for(pts3d, skeleton, template, alignment, articulation, fixed_body)

    positions, obs_weights = observations(pts3d, plan, weights)
    angles, root_pos, root_rot = _solve(
        quickik,
        plan,
        positions,
        obs_weights,
        n_iterations=n_iterations,
        neutral_weight=neutral_weight,
        damping=damping,
        position_tolerance=position_tolerance,
        angle_tolerance=angle_tolerance,
        parallel=parallel,
        segment_len=segment_len,
        overlap_len=overlap_len,
    )

    never_placed = frozenset(
        leg.name for leg in template.legs if leg.name not in alignment.leg_origin
    )
    for name in sorted(never_placed):
        log.warning(
            "inverse_kinematics: leg %r has no observed thorax-coxa; "
            "its angles are left unset",
            name,
        )
    unfittable = unfittable_branches(
        obs_weights, plan, never_placed=never_placed, absent_points=absent_points
    )
    for d, branch in enumerate(plan.dof_branch):
        bad = unfittable.get(branch)
        if bad is not None:
            angles[bad, d] = np.nan

    joints_model = plan.kinematics().joint_positions(angles, root_pos, root_rot)
    joints_world = plan.to_world(joints_model)
    model_pts3d = np.full((n_frames, n_points, 3), np.nan)
    rows = plan.joint_row
    tracked = rows >= 0
    model_pts3d[:, rows[tracked]] = joints_world[:, tracked]

    finite = int(np.isfinite(angles).all(axis=0).sum())
    log.info(
        "inverse kinematics: solved %d frames, %d/%d joint-angle tracks complete",
        n_frames,
        finite,
        angles.shape[1],
    )
    _warn_about_pinned_limits(angles, plan)
    return IKResult(
        angles=angles,
        angle_names=list(plan.angle_names),
        model_pts3d=model_pts3d,
        alignment=alignment,
        body_plan=plan,
        chain_scales=plan.chain_scales,
        chain_offsets=plan.chain_offsets,
        body_scale=float(plan.body_sim[1]),
    )


#: A DOF is "pinned" when it sits this close (radians) to one of its limits.
_PINNED_TOL = 1e-4

#: Report a DOF whose angle is pinned in at least this fraction of solved frames.
_PINNED_REPORT = 0.25


def _warn_about_pinned_limits(angles: np.ndarray, plan: BodyPlan) -> None:
    """Report DOFs whose joint limits, not the data, are deciding the fit.

    QuickIK enforces limits by clamping each Gauss-Newton step, without the gradient
    projection a trust-region method would apply, so a binding limit costs noticeably
    more accuracy than it used to. When a DOF spends most of the recording against a
    limit, that limit is what is capping the fit there -- worth saying out loud rather
    than leaving as an unexplained residual.

    The message deliberately does **not** say to widen it. A pinned limit is evidence that
    something is wrong, not that the limit is the wrong thing: a solver will happily drive
    one DOF into a wall to absorb a geometry error elsewhere in the chain, and both of the
    worst cases seen so far were exactly that. The legs pinned the +-50 degree thorax-coxa
    yaw wall in 71% of frames because two template axes were swapped and the leg frame was
    rotated wrongly -- once fixed, the same range pinned 0.7% of frames and the residual
    fell 41%, far more than widening had bought. The abdomen pinned its lateral wall in 54%
    of frames because two markers shared a body; correcting that stopped the pinning
    without touching the range. The way to tell the cases apart is to relax the limit and
    see whether the residual actually moves.
    """
    lo = np.array([d["limits"][0] for j in plan.plan["joints"] for d in j["dofs"]])
    hi = np.array([d["limits"][1] for j in plan.plan["joints"] for d in j["dofs"]])
    with np.errstate(all="ignore"):
        solved = np.isfinite(angles)
        at_limit = solved & (
            (np.abs(angles - lo) < _PINNED_TOL) | (np.abs(angles - hi) < _PINNED_TOL)
        )
        counts = solved.sum(axis=0)
        frac = np.divide(
            at_limit.sum(axis=0), counts, out=np.zeros(len(lo)), where=counts > 0
        )
    worst = np.argsort(-frac)
    pinned = [i for i in worst if frac[i] >= _PINNED_REPORT]
    if not pinned:
        return
    log.warning(
        "inverse kinematics: %d joint angle(s) sit at a limit in most frames, so the "
        "limits -- not the keypoints -- are capping the fit there. Check the chain's "
        "geometry before widening with [inverse_kinematics.bounds]: a pinned limit is "
        "often absorbing an error elsewhere, and then widening it does not help: %s",
        len(pinned),
        ", ".join(
            f"{plan.angle_names[i]} {frac[i]:.0%} "
            f"({np.degrees(lo[i]):.0f}..{np.degrees(hi[i]):.0f} deg)"
            for i in pinned[:6]
        )
        + (" ..." if len(pinned) > 6 else ""),
    )


# -- building the plan -------------------------------------------------------


def _plan_for(
    pts3d: np.ndarray,
    skeleton: Skeleton,
    template: KinematicTemplate,
    alignment: Alignment,
    articulation: Articulation | None,
    fixed_body: bool,
) -> BodyPlan:
    """Register the recording to the model and assemble its body plan.

    The registration is one similarity transform fit from the six thorax-coxa
    keypoints, which are body-fixed. It sets both the frame the plan is solved in and
    the recording's :attr:`~IKResult.body_scale`.
    """
    index = {name: i for i, name in enumerate(skeleton.point_names)}
    # The coxa reference lives with the baked articulation; load a chain-less copy when
    # no chains are being fit, so a legs-only run still registers to the model frame.
    reference = articulation if articulation is not None else Articulation.load(fit=())
    sim = _coxa_similarity(pts3d, index, reference)
    if sim is None:
        raise ValueError(
            "inverse_kinematics could not register the recording to the model: it "
            "needs at least three of the six thorax-coxa keypoints "
            f"({', '.join(reference.coxa_points)}) triangulated in some frame"
        )
    scales: dict[str, float] = {}
    offsets: dict[str, np.ndarray] = {}
    if articulation is not None:
        based = _chain_offsets(pts3d, index, articulation, sim)
        for name, shift in based.items():
            log.info(
                "inverse kinematics: %s chain placed on its measured base, %.3f model "
                "units off the registered one (%s)",
                name,
                float(np.linalg.norm(shift)),
                np.round(shift, 3).tolist(),
            )
        scales, offsets = _chain_calibration(
            pts3d, index, articulation, sim, based=based
        )
        if scales:
            log.info(
                "inverse kinematics: estimated chain scale %s",
                {k: round(v, 3) for k, v in scales.items()},
            )
        for name in sorted(set(offsets) - set(based)):
            log.info(
                "inverse kinematics: %s chain has no base landmark, so its root was "
                "FITTED from its whole marker set: %.3f model units off the registered "
                "one (%s)",
                name,
                float(np.linalg.norm(offsets[name])),
                np.round(offsets[name], 3).tolist(),
            )
    return build_body_plan(
        template,
        skeleton,
        alignment,
        sim,
        articulation=articulation,
        chain_scales=scales,
        chain_offsets=offsets,
        fixed_body=fixed_body,
    )


def _chain_offsets(
    pts3d: np.ndarray,
    index: dict[str, int],
    articulation: Articulation,
    sim: tuple[np.ndarray, float, np.ndarray],
) -> dict[str, np.ndarray]:
    """Each chain's measured base, as a model-frame shift from the model's own.

    A chain that names a base point (the head's ``neck``) is placed where that landmark
    was actually measured, the way each leg is placed on its measured median thorax-coxa
    -- and for the same reason. Left on the model's anchor, a chain's base is wherever
    the body registration put it, and that registration is a similarity fit to six
    thorax-coxa keypoints which are very nearly coplanar: on the standard rig their
    singular values run 0.693 / 0.325 / **0.085**, so the dorsal direction is barely
    determined, and the head pivot sits 0.404 above their centroid -- a 4.7x
    extrapolation along the worst-conditioned axis. Measured across a 55-recording
    corpus that lands the pivot a median 0.125 model units too dorsal, 29% of the head's
    own radius, with the same sign in every single recording. The antennae then absorb it
    as ~11 degrees of spurious pitch: the median fitted head pitch was 54.7 degrees
    against a +/- 60 limit, close enough to the ceiling that one recording spent 21% of
    its frames pinned there. Placed on the neck instead it reads 44.4 degrees, and
    nothing pins.

    The shift is a **median over the recording**, not per frame: the chain base is a
    body landmark on a tethered fly, so it is one point per recording and the median is
    what rejects the per-frame detection noise around it.

    The abdomen gets no shift *here*, and deliberately not the head's. Its root anchor sits
    at almost exactly the head pivot's dorsal height (1.300 against 1.301), which invites
    reusing the ``neck``'s measurement -- but the two anchors are on *opposite sides* of
    the coxa centroid in x (+0.53 and -0.36), and the registration's ill-determined mode
    is a pitch about that centroid, which tilts them opposite ways. Measured: searching
    the abdomen root over a grid at the fitted size, applying the neck's shift makes the
    marker residual **worse** than no shift in every recording tried (0.091 -> 0.149,
    0.082 -> 0.132, 0.061 -> 0.079, 0.226 -> 0.247), and that search put the abdomen's own
    optimum at *positive* dz.

    Its root is genuinely misplaced, and nothing in the *pose* measures it -- no keypoint
    sits on it. What does measure it is the chain's whole marker set, and
    :func:`_chain_calibration` now fits it there, jointly with the chain's size, for any
    chain this function leaves out. That is not a reversal of the paragraph above: the grid
    search held the size at the rigid-ruler value, which for the abdomen over-reads by
    10-19%, and size and root translation are coupled through an exact null direction of
    the fit -- so with the size free the optimum moves and its dz changes sign (to
    -0.176 / -0.137 / -0.107 on three animals, same sign each time). The objection that it
    would be "three free parameters trading against five angles" is also still true, and
    is why the fitted root is reduced to one value per recording rather than solved per
    frame.

    Returns ``{}`` unless a chain both names a base point and had it triangulated.
    """
    rot, scale, trans = sim
    out: dict[str, np.ndarray] = {}
    for chain in articulation.chains:
        col = chain.marker_index(chain.base_point or "")
        row = index.get(chain.base_point or "", -1)
        if col is None or row < 0:
            continue
        local = ((pts3d[:, row] - trans) @ rot) / max(scale, 1e-12)
        with np.errstate(all="ignore"):
            measured = np.nanmedian(local, axis=0)
        if not np.all(np.isfinite(measured)):
            log.warning(
                "inverse_kinematics: the %s chain's base point %r was never "
                "triangulated; leaving the chain on the registered base",
                chain.name,
                chain.base_point,
            )
            continue
        out[chain.name] = measured - np.asarray(chain.marker_neutral[col], dtype=float)
    return out


def _coxa_world(pts3d: np.ndarray, index: dict[str, int], reference: Articulation):
    """``(T, 6, 3)`` the thorax-coxa keypoints, NaN where the skeleton lacks one."""
    n_frames = pts3d.shape[0]
    cols = [index.get(p, -1) for p in reference.coxa_points]
    return np.stack(
        [pts3d[:, c] if c >= 0 else np.full((n_frames, 3), np.nan) for c in cols],
        axis=1,
    )


def _coxa_similarity(pts3d: np.ndarray, index: dict[str, int], reference: Articulation):
    """``(R, s, t)`` mapping the model onto the recording, from the median coxae."""
    world = _coxa_world(pts3d, index, reference)
    with np.errstate(all="ignore"):
        measured = np.nanmedian(world, axis=0)  # (6, 3)
    return body_similarity(reference.coxa_neutral, measured)


def _chain_markers_local(
    pts3d: np.ndarray,
    index: dict[str, int],
    chain,
    sim: tuple[np.ndarray, float, np.ndarray],
) -> np.ndarray:
    """``(T, M, 3)`` a chain's measured markers, read in the model frame (NaN if absent)."""
    rot, scale, trans = sim
    n_frames = pts3d.shape[0]
    cols = [index.get(name, -1) for name in chain.marker_names]
    world = np.stack(
        [pts3d[:, c] if c >= 0 else np.full((n_frames, 3), np.nan) for c in cols],
        axis=1,
    )
    return ((world - trans) @ rot) / max(scale, 1e-12)


def _chain_calibration(
    pts3d: np.ndarray,
    index: dict[str, int],
    articulation: Articulation,
    sim: tuple[np.ndarray, float, np.ndarray],
    *,
    based: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Each chain's size and root, fitted against its whole marker set.

    Size and root are estimated **together** and per chain, because they are coupled: the
    chain's first two hinges and a translation of its root trade along an exact null
    direction of the fit, so measuring either with the other held at a wrong value gives a
    wrong -- and confidently repeatable -- answer for both. See
    :func:`~deeperfly.inverse_kinematics.articulation.calibrate_chain` for the measurement
    and what it is worth on each chain.

    A chain that :func:`_chain_offsets` already placed on a base landmark (``based``) keeps
    that root and only has its size fitted; one with no landmark has both fitted. Both come
    back as per-recording constants, which is what leaves the per-frame angle solve free of
    that null direction.
    """
    based = dict(based or {})
    scales: dict[str, float] = {}
    offsets: dict[str, np.ndarray] = dict(based)
    for chain in articulation.chains:
        local = _chain_markers_local(pts3d, index, chain, sim)
        # Only a seed: `calibrate_chain` below measures the size without needing an
        # articulation-invariant pair, and the abdomen deliberately no longer has one.
        ruler = estimate_chain_scale(local, chain, warn_if_unmeasurable=False)
        fit_root = chain.name not in based
        scale, shift = calibrate_chain(
            local,
            chain,
            seed_scale=ruler,
            base_shift=based.get(chain.name),
            fit_root=fit_root,
        )
        scales[chain.name] = scale
        if fit_root:
            offsets[chain.name] = shift
        if abs(scale - ruler) > 0.05 * max(ruler, 1e-9):
            log.info(
                "inverse_kinematics: the %s chain's whole-marker size %.4f differs from "
                "its rigid-ruler estimate %.4f by %+.1f%%; the ruler is believed only "
                "where it has a long, well-observed baseline (see calibrate_chain)",
                chain.name,
                scale,
                ruler,
                100.0 * (scale / max(ruler, 1e-9) - 1.0),
            )
    return scales, offsets


# -- observations ------------------------------------------------------------


def observations(
    pts3d: np.ndarray, plan: BodyPlan, weights: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Gather the plan's observations: ``(T, N, 3)`` model positions and ``(T, N)`` weights.

    A joint that tracks no keypoint (the root, the head/abdomen hinges), a NaN position
    and a non-positive or non-finite weight are all "missing" to QuickIK, which reads
    that from the weight alone -- so the position of a missing observation is filled with
    zero rather than left NaN.
    """
    n_frames = pts3d.shape[0]
    rows = plan.joint_row
    tracked = rows >= 0

    model = plan.to_model(pts3d)  # (T, P, 3)
    positions = np.zeros((n_frames, plan.n_joints, 3))
    positions[:, tracked] = model[:, rows[tracked]]

    obs = np.zeros((n_frames, plan.n_joints))
    if weights is None:
        obs[:, tracked] = 1.0
    else:
        w = np.asarray(weights, dtype=float)
        obs[:, tracked] = np.where(
            np.isfinite(w[:, rows[tracked]]), w[:, rows[tracked]], 0.0
        )
    obs[~np.isfinite(positions).all(axis=-1)] = 0.0
    positions[~np.isfinite(positions)] = 0.0
    return positions, np.clip(obs, 0.0, None)


def unfittable_branches(
    obs_weights: Float[np.ndarray, "T N"],
    plan: BodyPlan,
    *,
    never_placed: frozenset[str] = frozenset(),
    absent_points: Bool[np.ndarray, "T P"] | None = None,
) -> dict[str, np.ndarray]:
    """``branch -> (T,) bool`` frames whose observations cannot support a fit.

    Re-imposes what a per-limb solver got for free: a limb needs at least
    :data:`_MIN_VALID_JOINTS` observed keypoints in a frame to mean anything, and a leg
    named in ``never_placed`` (its coxa was never seen, so it has no measured position at
    all) is not fitted in any frame. QuickIK has no notion of an unconstrained sub-chain
    -- it returns the neutral-biased answer for every DOF regardless -- so this is where
    "not fitted" becomes NaN again, for the batch fit and the editor's live re-fit alike.

    ``absent_points`` is the ``(T, P)`` "not on this animal" declaration (a ``(P,)``
    whole-recording one is accepted and broadcast). It lowers a branch's threshold to what
    that branch *can* still deliver **in that frame**, because an anatomically absent
    keypoint is categorically different from one the detector merely missed -- and because
    a leg lost part-way through should be fitted in full before the loss and as a stump
    after:

    - the ``head`` chain has three DOFs but exactly two markers (the antennae). Declaring
      one antenna absent -- a routine unilateral ablation -- would otherwise put the head
      permanently below a flat threshold of 2 and NaN all three head DOFs forever.
    - an amputated leg leaves a stump whose remaining joints are real and worth measuring,
      which is precisely what a leg-loss study exists to record.

    A missing *detection* must NOT lower the bar the same way: a leg the detector found
    once is a tracking failure, and reporting neutral-biased angles for it would be a
    fabricated measurement. So the threshold is computed from the declaration and the
    plan's topology only, never from what happened to be observed.

    Lowering the bar is itself capped by whether the remaining markers can *determine* the
    branch: each observed joint contributes three coordinates, so the threshold only drops
    below :data:`_MIN_VALID_JOINTS` while ``3 * n_fittable >= n_dofs``. One antenna (3
    coordinates, 3 head DOFs) qualifies; a leg amputated down to a single coxa (3
    coordinates, 7 DOFs) does not, and stays unfittable rather than reporting the
    neutral-biased answer QuickIK would return for it.

    A chain's **base landmark** is not counted at all, however well observed it is
    (:attr:`~deeperfly.inverse_kinematics.bodyplan.BodyPlan.joint_is_base`). The head's
    ``neck`` sits on the rotation center the three head DOFs turn about, so no angle can
    move it and its residual carries no gradient; counting it would let the head pass
    this test on evidence that cannot constrain a single one of its angles. It is why
    declaring both antennae absent leaves the head unfitted rather than "fitted" from
    the neck alone.
    """
    branches = np.asarray(plan.joint_branch)
    # A joint can contribute an observation only if it maps to a keypoint at all
    # (`joint_row < 0` is a pure kinematic joint, e.g. two of the three head DOFs), it is
    # not its chain's base landmark, and that keypoint is on this animal.
    n_frames = np.asarray(obs_weights).shape[0]
    rows = np.asarray(plan.joint_row)
    tracked = (rows >= 0) & ~plan.joint_is_base
    observed = (np.asarray(obs_weights) > 0) & tracked[None, :]
    fittable_joint = np.broadcast_to(tracked, (n_frames, tracked.size)).copy()
    if absent_points is not None:
        absent = np.asarray(absent_points, dtype=bool)
        if absent.ndim == 1:  # a whole-recording declaration broadcasts over time
            absent = np.broadcast_to(absent.reshape(1, -1), (n_frames, absent.size))
        # (T, N): for each joint that maps to a keypoint, is that keypoint on the animal
        # in this frame?
        present = ~absent[:, np.clip(rows, 0, absent.shape[1] - 1)]
        fittable_joint &= np.where(tracked[None, :], present, False)
    dof_branches = np.asarray(plan.dof_branch)
    out: dict[str, np.ndarray] = {}
    for branch in sorted(set(plan.dof_branch)):
        if not branch:
            continue
        in_branch = branches == branch
        n_fittable = fittable_joint[:, in_branch].sum(axis=1)  # (T,)
        n_dofs = int((dof_branches == branch).sum())
        count = observed[:, in_branch].sum(axis=1)  # (T,)
        # Relax the bar only as far as the frame's own remaining markers can determine the
        # branch: each observed joint contributes three coordinates, so the threshold drops
        # below _MIN_VALID_JOINTS only while 3 * n_fittable >= n_dofs. One antenna (3
        # coordinates, 3 head DOFs) qualifies; a leg amputated to a single coxa (3
        # coordinates, 7 DOFs) does not.
        determinable = (n_fittable >= _MIN_VALID_JOINTS) | (3 * n_fittable >= n_dofs)
        threshold = np.minimum(_MIN_VALID_JOINTS, n_fittable)
        # n_fittable == 0 is the genuine "nothing to fit" case: lowering the bar to zero
        # would report neutral-pose angles as if they were measured.
        bad = np.where((n_fittable == 0) | ~determinable, True, count < threshold)
        if branch in never_placed:
            bad = np.ones_like(bad)
        if bad.any():
            out[branch] = bad
    return out


# -- the solve ---------------------------------------------------------------


def _solve(
    quickik,
    plan: BodyPlan,
    positions: np.ndarray,
    weights: np.ndarray,
    *,
    n_iterations: int,
    neutral_weight: float,
    damping: float,
    position_tolerance: float,
    angle_tolerance: float,
    parallel: bool,
    segment_len: int,
    overlap_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run QuickIK over the sequence and return ``(angles, root_pos, root_rot)``.

    Sequential by default: :class:`quickik.SequenceSolver` warm-starts each frame from
    the previous one, which both speeds up convergence and is the only thing giving the
    angle traces temporal continuity. The segmented-parallel path restarts every segment
    from the neutral pose, so it trades a seam in the traces for wall-clock.
    """
    tree = quickik.KinematicTree.from_json_str(plan.to_json())
    config = quickik.SolverConfig(
        n_iterations=int(n_iterations),
        neutral_weight=float(neutral_weight),
        position_tolerance=float(position_tolerance),
        angle_tolerance=float(angle_tolerance),
        damping=float(damping),
    )
    n_frames = positions.shape[0]
    if parallel and n_frames > segment_len:
        log.info(
            "inverse kinematics: solving %d frames in parallel segments of %d "
            "(overlap %d); expect small steps at the segment seams",
            n_frames,
            segment_len,
            overlap_len,
        )
        poses = quickik.solve_sequence_segmented_parallel(
            tree,
            config,
            positions,
            weights,
            quickik.ParallelSolveConfig(
                segment_len=int(segment_len),
                overlap_len=int(overlap_len),
                overlap_tolerance=0.05,
                n_workers=-1,
            ),
        )
    else:
        poses = quickik.SequenceSolver(tree, config).solve_sequence(positions, weights)

    angles = np.asarray([p.dof_angles for p in poses], dtype=float)
    root_pos = np.asarray([p.root_pos for p in poses], dtype=float)
    root_rot = np.asarray([p.root_rot for p in poses], dtype=float)
    return angles, root_pos, root_rot
