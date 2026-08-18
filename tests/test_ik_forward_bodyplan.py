"""Tests for the numpy forward kinematics and the generated QuickIK body plan.

Deliberately free of any QuickIK import: these cover the geometry deeperfly owns --
building the body plan and evaluating it -- which must keep working on an install
without the optional solver (the mesh overlay and the GUI's static-fit path depend on
exactly this code).

The load-bearing test here is
:func:`test_plan_fk_matches_the_mesh_node_convention`: it is what guarantees the angles
QuickIK fits on the generated plan and the head/abdomen mesh nodes the overlay poses
from :func:`~deeperfly.inverse_kinematics.forward.chain_affine` describe the *same*
transform. Everything else in the overlay's consistency follows from it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from helpers import (
    IK_BASELINE_PATH,
    fly38_skeleton,  # noqa: F401
)

from deeperfly.config import Config
from deeperfly.inverse_kinematics import forward as fwd
from deeperfly.inverse_kinematics.align import body_alignment
from deeperfly.inverse_kinematics.articulation import body_similarity, load_articulation
from deeperfly.inverse_kinematics.bodyplan import (
    PLAN_VERSION,
    ROOT_NAME,
    BodyPlan,
    _model_body_axes,
    _model_seglens,
    build_body_plan,
)
from deeperfly.inverse_kinematics.mesh import load_nmf_mesh
from deeperfly.inverse_kinematics.template import KinematicTemplate
from deeperfly.skeleton import Skeleton

_SEGLENS = np.array([0.0, 0.40, 0.69, 0.54, 0.63])


@pytest.fixture(scope="module")
def fly() -> Skeleton:
    return fly38_skeleton()


@pytest.fixture(scope="module")
def template() -> KinematicTemplate:
    return KinematicTemplate.load("neuromechfly")


@pytest.fixture(scope="module")
def articulation():
    return load_articulation()


@pytest.fixture(scope="module")
def real_pts3d() -> np.ndarray:
    """The example recording's triangulated pose, from the baseline fixture."""
    with np.load(IK_BASELINE_PATH, allow_pickle=True) as z:
        return z["real_pts3d"]


def _measure(real_pts3d, fly, template):
    """``(alignment, body_sim)`` measured from the real pose, as the stage does.

    ``real_pts3d`` is stored in ``fly38`` order, so it is re-gathered **by name** for
    whichever skeleton is asked for; a point the fixture has no column for stays NaN.
    """
    articulation = load_articulation()
    src = {n: i for i, n in enumerate(fly38_skeleton().point_names)}
    pts3d = np.stack(
        [
            real_pts3d[:, src[n]]
            if n in src
            else np.full((real_pts3d.shape[0], 3), np.nan)
            for n in fly.point_names
        ],
        axis=1,
    )
    align = body_alignment(pts3d, fly, template)
    index = {n: i for i, n in enumerate(fly.point_names)}
    coxae = np.stack([pts3d[:, index[p]] for p in articulation.coxa_points], axis=1)
    with np.errstate(all="ignore"):
        sim = body_similarity(articulation.coxa_neutral, np.nanmedian(coxae, axis=0))
    assert sim is not None
    return align, sim


@pytest.fixture(scope="module")
def measurements(real_pts3d, fly, template):
    return _measure(real_pts3d, fly, template)


def make_plan(fly, template, articulation, measurements, **kw) -> BodyPlan:
    align, sim = measurements
    return build_body_plan(template, fly, align, sim, articulation=articulation, **kw)


@pytest.fixture(scope="module")
def plan(fly, template, articulation, measurements) -> BodyPlan:
    return make_plan(fly, template, articulation, measurements)


# -- forward kinematics -------------------------------------------------------


def test_leg_fk_rest_pose_points_straight_down(template):
    """With every angle zero a leg is straight along its own frame's ``-z``.

    The template's stated rest pose. Kept as a standalone check on ``leg_fk`` because it
    is the invariant that catches a sign or axis-order slip in the chain, independently of
    the body plan built on top of it.
    """
    for leg in template.legs:
        joints = fwd.leg_fk(
            np.zeros(sum(leg.dof_counts)), leg.axes, _SEGLENS, leg.dof_counts
        )
        np.testing.assert_allclose(joints[:, :2], 0.0, atol=1e-12)
        np.testing.assert_allclose(joints[:, 2], -np.cumsum(_SEGLENS), atol=1e-12)


def test_chain_fk_agrees_with_chain_affine(articulation):
    """The vectorized chain FK and the per-depth affine describe one transform.

    ``chain_fk`` walks the markers in one pass; ``chain_affine`` is what the mesh overlay
    calls per node. They must not drift apart -- that would put the fitted angles and the
    mesh drawn from them in different poses.
    """
    rng = np.random.default_rng(1)
    for chain in articulation.chains:
        theta = rng.uniform(*chain.bounds)
        got = fwd.chain_fk(
            chain.anchors, chain.axes, chain.marker_depth, theta, chain.marker_neutral
        )
        for k, depth in enumerate(chain.marker_depth):
            a, b = fwd.chain_affine(chain, int(depth), theta)
            np.testing.assert_allclose(
                got[k], a @ chain.marker_neutral[k] + b, rtol=0, atol=1e-12
            )


def test_chain_affine_re_export_is_the_same_function(articulation):
    """``articulation.chain_affine`` still resolves, to the implementation in ``forward``."""
    from deeperfly.inverse_kinematics import articulation as art_mod

    assert art_mod.chain_affine is fwd.chain_affine


def test_rest_pose_leaves_a_baked_chain_at_its_neutral_markers(articulation):
    """All-zero chain angles put every marker back at its neutral model position."""
    for chain in articulation.chains:
        got = fwd.chain_fk(
            chain.anchors,
            chain.axes,
            chain.marker_depth,
            np.zeros(len(chain.dof_names)),
            chain.marker_neutral,
        )
        np.testing.assert_allclose(got, chain.marker_neutral, rtol=0, atol=1e-12)


# -- quaternion helpers -------------------------------------------------------


def test_quaternion_round_trips_including_half_turns():
    """``rmat_to_quat``/``quat_to_rmat`` invert each other, 180 degrees included.

    The half-turn is the case a naive ``w``-first formula loses (``w -> 0``), and it is
    reachable here: a body frame can be a half turn from the model's.
    """
    rng = np.random.default_rng(2)
    for _ in range(200):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rmat = fwd.axis_rmat(axis, rng.uniform(-np.pi, np.pi))
        np.testing.assert_allclose(
            fwd.quat_to_rmat(fwd.rmat_to_quat(rmat)), rmat, rtol=0, atol=1e-12
        )
    for axis in np.eye(3):
        rmat = fwd.axis_rmat(axis, np.pi)
        np.testing.assert_allclose(
            fwd.quat_to_rmat(fwd.rmat_to_quat(rmat)), rmat, rtol=0, atol=1e-12
        )


def test_axis_rmat_batches_over_angles():
    """A batch of angles gives a batch of rotations, matching the scalar form."""
    axis = np.array([0.0, 1.0, 0.0])
    angles = np.linspace(-1.0, 1.0, 7)
    batched = fwd.axis_rmat(axis, angles)
    assert batched.shape == (7, 3, 3)
    for k, angle in enumerate(angles):
        np.testing.assert_allclose(batched[k], fwd.axis_rmat(axis, angle), atol=0)


# -- the generated body plan --------------------------------------------------


def test_plan_shape_and_coverage(plan, fly, template, articulation):
    """The plan holds one joint per model point plus the hinges that carry no keypoint."""
    n_chain_dofs = sum(len(c.dof_names) for c in articulation.chains)
    n_markers = sum(len(c.marker_names) for c in articulation.chains)
    n_leg_joints = sum(len(leg.joints) for leg in template.legs)
    assert plan.n_joints == 1 + n_leg_joints + n_chain_dofs + n_markers
    assert plan.n_dofs == len(template.dof_names) + n_chain_dofs

    planned = {p for p in plan.joint_point if p}
    tracked = {p for p, r in zip(plan.joint_point, plan.joint_row) if r >= 0}
    assert tracked == planned & set(fly.point_names)
    assert int((plan.joint_row >= 0).sum()) == len(tracked)
    assert plan.joint_names[0] == ROOT_NAME
    assert plan.joint_point[0] is None and plan.joint_branch[0] == ""


@pytest.mark.parametrize(
    "name,model_only,skeleton_only",
    [
        ("fly38b", [], []),
        (
            "fly38",
            ["abdomen0", "abdomen1", "abdomen2", "abdomen3", "abdomen4", "neck"],
            [
                "l_abdomen0",
                "l_abdomen1",
                "l_abdomen2",
                "r_abdomen0",
                "r_abdomen1",
                "r_abdomen2",
            ],
        ),
    ],
)
def test_which_points_the_plan_covers(
    name, model_only, skeleton_only, template, articulation, real_pts3d
):
    """Coverage is an intersection, and both gaps are named rather than papered over.

    A joint tracks a keypoint only where the *model* has a marker for it and the *run's
    skeleton* names it, so the two sets are allowed to disagree. On ``fly38b`` they no
    longer do: with the head and abdomen chains both targeted at it, the plan covers all
    38 points and the fit has nothing left un-modelled.

    ``fly38`` is the other side of that. Its two historical abdomen side chains have no
    marker on the model any more and its skeleton has no ``neck``, so eleven points fall
    out on one side or the other -- the whole abdomen among them, which means a ``fly38``
    run gets no abdomen fit at all. Asserted by name so a future retarget fails this test
    rather than passing it quietly.
    """
    fly = Skeleton.from_config(Config.from_dict({"skeleton": {"name": name}}))
    plan = make_plan(fly, template, articulation, _measure(real_pts3d, fly, template))
    planned = {p for p in plan.joint_point if p}
    assert sorted(planned - set(fly.point_names)) == model_only
    assert sorted(set(fly.point_names) - planned) == skeleton_only


def test_plan_angle_names_match_the_results_contract(plan):
    """The plan's DOF order reproduces the ``angle_names`` written to ``results.h5``.

    Legs (in config order) then head then abdomen, with the flygym
    ``<parent>-<child>-<dof>`` names -- the contract the mesh overlay's by-name angle
    lookups and every downstream consumer read.
    """
    with np.load(IK_BASELINE_PATH, allow_pickle=True) as z:
        recorded = [str(n) for n in z["real_angle_names"]]
    assert list(plan.angle_names) == recorded


def test_plan_respects_the_configured_leg_subset(fly, articulation, measurements):
    """``legs = [...]`` restricts the plan and keeps the configured order."""
    template = KinematicTemplate.load("neuromechfly", legs=["rf", "lf"])
    plan = make_plan(fly, template, articulation, measurements)
    legs = [b for b in plan.joint_branch if b in {"rf", "lf"}]
    assert legs[0] == "rf" and legs[-1] == "lf"
    assert plan.angle_names[0].startswith("c_thorax-rf_coxa")
    assert not any(b in {"lm", "lh", "rm", "rh"} for b in plan.joint_branch)


def test_plan_is_strictly_json_round_trippable(plan):
    """The plan serializes with no ``default=`` fallback and reloads identically.

    ``json.dumps(..., default=str)`` would render a stray numpy array as the *string*
    ``"[1. 2.]"`` -- precision and type gone, no exception -- so the plan must be pure
    Python before it is ever written to ``results.h5``.
    """
    text = plan.to_json()
    assert json.loads(text) == plan.plan
    assert json.dumps(plan.plan)  # no default= : raises on any stray numpy value


def test_plan_json_stays_well_under_the_hdf5_attribute_cap(plan):
    """A size guard: the plan is stored as a dataset, but keep it small regardless."""
    assert len(plan.to_json()) < 64 * 1024


def test_plan_records_its_version_and_registration(plan):
    """The plan carries what a reader needs to interpret it, under ``x-deeperfly``."""
    meta = plan.plan["x-deeperfly"]
    assert meta["version"] == PLAN_VERSION
    assert meta["template"] == "neuromechfly"
    assert np.allclose(np.asarray(meta["body_sim"]["rot"]), plan.body_sim[0])
    assert meta["body_sim"]["scale"] == pytest.approx(plan.body_sim[1])


def test_plan_rebuilt_from_json_matches_the_original(plan, fly):
    """``BodyPlan.from_json`` recovers the bookkeeping, so the GUI reuses the fit's plan."""
    again = BodyPlan.from_json(plan.to_json(), fly)
    assert again.joint_names == plan.joint_names
    assert again.joint_point == plan.joint_point
    assert again.joint_branch == plan.joint_branch
    assert again.angle_names == plan.angle_names
    assert again.dof_branch == plan.dof_branch
    assert again.fixed_base == plan.fixed_base
    np.testing.assert_array_equal(again.joint_row, plan.joint_row)
    np.testing.assert_allclose(again.neutral, plan.neutral, atol=0)
    np.testing.assert_allclose(again.body_sim[0], plan.body_sim[0], atol=0)
    assert again.body_sim[1] == pytest.approx(plan.body_sim[1])
    np.testing.assert_allclose(again.body_sim[2], plan.body_sim[2], atol=0)
    assert again.chain_scales == pytest.approx(plan.chain_scales)


def test_plan_neutral_is_mid_range_not_the_singular_zero_pose(plan, template):
    """Every DOF's neutral sits inside its limits, away from the straight-leg pose.

    A leg at all-zero angles is straight, where the Jacobian's null direction is pure
    thorax-coxa yaw; QuickIK is plain damped Gauss-Newton with no line search, so
    starting there (and pulling back toward there) leaves large residual angle error.
    The abdomen has the mirror-image problem: its limits are ``[-30, 0]`` degrees, so a
    zero neutral would sit exactly *on* a bound and flatten the curl.
    """
    col = {n: i for i, n in enumerate(plan.angle_names)}
    for joint in plan.plan["joints"]:
        for dof in joint["dofs"]:
            lo, hi = dof["limits"]
            assert lo <= dof["neutral"] <= hi
            if lo < hi:
                assert lo < dof["neutral"] < hi
    for leg in template.legs:
        lo, hi = leg.bounds
        got = np.array([plan.neutral[col[n]] for n in leg.dof_names])
        np.testing.assert_allclose(got, np.clip(0.5 * (lo + hi), lo, hi), atol=1e-12)


def test_plan_frames_round_trip_between_model_and_world(plan, real_pts3d):
    """``to_model``/``to_world`` invert each other on the recording's own points."""
    np.testing.assert_allclose(
        plan.to_world(plan.to_model(real_pts3d)), real_pts3d, rtol=0, atol=1e-9
    )


# -- evaluating the plan ------------------------------------------------------


def test_plan_leg_rest_pose_points_down_the_model_body_axis(
    plan, template, fly, measurements
):
    """At all-zero angles each leg is straight along the model body frame's ``-z``.

    The invariant that catches a sign or axis error in the conversion: the template's
    rest pose is a straight leg pointing down its *own* frame's ``-z``, and the plan
    carries that frame on the thorax-coxa's ``offset_quat``.
    """
    align, sim = measurements
    down = _model_body_axes() @ np.array([0.0, 0.0, -1.0])
    positions = plan.kinematics().joint_positions(np.zeros(plan.n_dofs))[0]
    row = {name: i for i, name in enumerate(plan.joint_names)}
    for leg in template.legs:
        seglens = np.cumsum(_model_seglens(leg, fly, align, sim[1]))
        base = positions[row[leg.joints[0].point]]
        for j, joint in enumerate(leg.joints):
            np.testing.assert_allclose(
                positions[row[joint.point]], base + down * seglens[j], atol=1e-12
            )


def test_plan_rest_pose_places_markers_at_their_neutral(plan, articulation):
    """At all-zero angles the head/abdomen markers sit at their neutral model points."""
    positions = plan.kinematics().joint_positions(np.zeros(plan.n_dofs))[0]
    row = {name: i for i, name in enumerate(plan.joint_names)}
    for chain in articulation.chains:
        for m, point in enumerate(chain.marker_names):
            np.testing.assert_allclose(
                positions[row[point]], chain.marker_neutral[m], atol=1e-12
            )


@pytest.mark.parametrize(
    "scales",
    [
        {"head": 1.0, "abdomen": 1.0},
        {"head": 1.7, "abdomen": 0.7},
        {"head": 0.8, "abdomen": 1.43},
    ],
)
def test_plan_fk_matches_the_mesh_node_convention(
    fly, template, articulation, measurements, scales
):
    """The plan's chains and the mesh overlay's nodes are the same transform.

    The overlay poses a head/abdomen node mesh by ``chain_affine`` at that node's chain
    depth and then grows it about the chain's base anchor by the data-estimated size
    (``NmfMesh._node_transforms``). The plan bakes that same growth into its chain
    offsets. If these ever disagree, the fitted angles and the mesh drawn from them
    describe different poses -- which is exactly what the pre-QuickIK solver did (it
    scaled the *markers* about the base while leaving the anchors at model size,
    diverging by ~20% of the abdomen's span at the deepest marker).
    """
    plan = make_plan(fly, template, articulation, measurements, chain_scales=scales)
    rng = np.random.default_rng(7)
    col = {name: i for i, name in enumerate(plan.angle_names)}
    row = {name: i for i, name in enumerate(plan.joint_names)}
    angles = np.zeros(plan.n_dofs)
    for chain in articulation.chains:
        for k, name in enumerate(chain.dof_names):
            angles[col[name]] = rng.uniform(*[b[k] for b in chain.bounds])
    positions = plan.kinematics().joint_positions(angles)[0]

    for chain in articulation.chains:
        f = scales[chain.name]
        base = np.asarray(chain.anchors[0], dtype=float)
        theta = np.array([angles[col[n]] for n in chain.dof_names])
        for m, point in enumerate(chain.marker_names):
            a, b = fwd.chain_affine(chain, int(chain.marker_depth[m]), theta)
            neutral = np.asarray(chain.marker_neutral[m], dtype=float)
            as_mesh_poses_it = f * (a @ neutral + b) + (1.0 - f) * base
            np.testing.assert_allclose(
                positions[row[point]], as_mesh_poses_it, rtol=0, atol=1e-12
            )


def test_plan_fk_matches_chain_affine_at_every_mesh_node_depth(
    plan, articulation, measurements
):
    """Every ``(chain, depth)`` the baked mesh asset asks for is reproduced by plan FK.

    ``nmf_mesh.npz`` names the node slots the overlay poses; two abdomen depths carry
    no marker at all, so a check that only walked the markers would miss them.
    """
    mesh = load_nmf_mesh()
    wanted = {
        (int(c), int(d))
        for c, d in zip(mesh.slot_chain, mesh.slot_depth)
        if c >= 0 and d >= 0
    }
    assert wanted, "the packaged mesh asset should have articulated node slots"
    rng = np.random.default_rng(11)
    col = {name: i for i, name in enumerate(plan.angle_names)}
    angles = np.zeros(plan.n_dofs)
    for chain in articulation.chains:
        for k, name in enumerate(chain.dof_names):
            angles[col[name]] = rng.uniform(*[b[k] for b in chain.bounds])

    kin = plan.kinematics()
    row = {name: i for i, name in enumerate(plan.joint_names)}
    positions = kin.joint_positions(angles)[0]
    for chain_idx, depth in sorted(wanted):
        chain = articulation.chains[chain_idx]
        assert depth <= len(chain.dof_names)
        theta = np.array([angles[col[n]] for n in chain.dof_names])
        a, b = fwd.chain_affine(chain, depth, theta)
        # A depth-d node is carried by the plan's joint d-1 (or the root at depth 0);
        # probe it with the anchor of that joint, whose image is (A c + b).
        anchor = np.asarray(chain.anchors[depth - 1 if depth else 0], dtype=float)
        if depth == 0:
            np.testing.assert_allclose(a, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(b, np.zeros(3), atol=1e-12)
            continue
        np.testing.assert_allclose(
            positions[row[chain.dof_names[depth - 1]]],
            a @ np.asarray(chain.anchors[depth - 1], dtype=float) + b,
            rtol=0,
            atol=1e-12,
        )
        assert np.isfinite(anchor).all()


def test_plan_fk_batches_over_frames(plan):
    """A batch of poses evaluates to a batch of positions, frame by frame identical."""
    rng = np.random.default_rng(13)
    angles = rng.uniform(-0.2, 0.2, size=(5, plan.n_dofs))
    kin = plan.kinematics()
    batched = kin.joint_positions(angles)
    assert batched.shape == (5, plan.n_joints, 3)
    for t in range(5):
        np.testing.assert_allclose(
            batched[t], kin.joint_positions(angles[t])[0], atol=0
        )


def test_plan_fk_propagates_nan_angles_distally(plan, template):
    """A NaN angle NaNs its joint's descendants and leaves the rest finite.

    How an unfittable limb stays visibly absent from the overlay instead of being drawn
    at some invented pose.
    """
    kin = plan.kinematics()
    row = {name: i for i, name in enumerate(plan.joint_names)}
    col = {name: i for i, name in enumerate(plan.angle_names)}
    leg = template.legs[0]
    angles = np.zeros(plan.n_dofs)
    angles[col[leg.dof_names[0]]] = np.nan  # a thorax-coxa DOF: everything below it
    positions = kin.joint_positions(angles)[0]
    assert np.isfinite(positions[row[leg.joints[0].point]]).all()  # the joint itself
    for joint in leg.joints[1:]:
        assert not np.isfinite(positions[row[joint.point]]).any()
    other = template.legs[1]
    for joint in other.joints:
        assert np.isfinite(positions[row[joint.point]]).all()


def test_plan_falls_back_to_a_model_length_for_an_unmeasured_segment(
    fly, template, articulation, measurements
):
    """A never-observed bone takes the model's own length, not a zero-length link.

    ``align`` reports ``0.0`` for a segment it never saw; a zero-length link's DOFs
    would have no moment arm at all (a column of zeros in the Jacobian).
    """
    align, sim = measurements
    leg = template.legs[0]
    seglens = dict(align.seglens)
    broken = np.array(seglens[leg.name], dtype=float)
    broken[2] = 0.0
    seglens[leg.name] = broken
    from dataclasses import replace

    lens = _model_seglens(leg, fly, replace(align, seglens=seglens), sim[1])
    a = load_nmf_mesh().kp_neutral[fly.point_names.index(leg.point_names[1])]
    b = load_nmf_mesh().kp_neutral[fly.point_names.index(leg.point_names[2])]
    assert lens[2] == pytest.approx(float(np.linalg.norm(b - a)))
    assert lens[1] > 0 and lens[3] > 0


# -- malformed plans ----------------------------------------------------------


def test_plan_kinematics_rejects_a_plan_with_no_single_root():
    """Two roots (or none) is a structural error, not something to guess at."""
    joint = {
        "name": "a",
        "parent": None,
        "offset_pos": [0.0, 0.0, 0.0],
        "offset_quat": [1.0, 0.0, 0.0, 0.0],
        "dofs": [],
    }
    two_roots = {"fixed_base": True, "joints": [joint, {**joint, "name": "b"}]}
    with pytest.raises(ValueError, match="exactly one root joint"):
        fwd.PlanKinematics.from_plan(two_roots)


def test_plan_kinematics_rejects_an_unknown_parent():
    joints = [
        {
            "name": "a",
            "parent": None,
            "offset_pos": [0.0, 0.0, 0.0],
            "offset_quat": [1.0, 0.0, 0.0, 0.0],
            "dofs": [],
        },
        {
            "name": "b",
            "parent": "nope",
            "offset_pos": [0.0, 0.0, 0.0],
            "offset_quat": [1.0, 0.0, 0.0, 0.0],
            "dofs": [],
        },
    ]
    with pytest.raises(ValueError, match="unknown parent"):
        fwd.PlanKinematics.from_plan({"fixed_base": True, "joints": joints})


def test_plan_kinematics_rejects_the_wrong_number_of_angles(plan):
    with pytest.raises(ValueError, match="DOF angles per frame"):
        plan.kinematics().joint_positions(np.zeros(plan.n_dofs + 1))


# -- placing a chain on its measured base -------------------------------------


def test_chain_offset_moves_only_the_chain_root_and_its_root_parented_markers(
    fly, template, articulation, measurements
):
    """A base shift is rigid: it moves the two absolute offsets and nothing else.

    Every other offset in a chain is a *difference* between two neutral anchors, which a
    rigid translation leaves alone -- so a shift must not change the chain's internal
    geometry, only where the whole thing sits. That is what makes it composable with the
    size estimate, which scales those differences.
    """
    shift = np.array([0.03, -0.02, -0.13])
    base = make_plan(fly, template, articulation, measurements)
    moved = make_plan(
        fly, template, articulation, measurements, chain_offsets={"head": shift}
    )
    by_name = {j["name"]: j for j in base.plan["joints"]}
    head = articulation.chain("head")
    root_parented = {head.dof_names[0], "neck"}  # the hinge stack's root, and the base

    for j in moved.plan["joints"]:
        want = np.asarray(by_name[j["name"]]["offset_pos"], dtype=float)
        if j["x-deeperfly-branch"] == "head" and j["name"] in root_parented:
            want = want + shift
        np.testing.assert_allclose(j["offset_pos"], want, atol=1e-12)


def test_chain_offsets_round_trip_through_the_stored_plan(
    fly, template, articulation, measurements
):
    """The editor's live re-fit must re-solve on the pivot the pipeline fitted about.

    It rebuilds the plan from the JSON in ``results.h5`` rather than re-deriving it, so
    the shift has to survive that trip -- otherwise a corrected label would be fitted
    about the model's anchor while the stored angles describe the measured one.
    """
    shift = np.array([0.03, -0.02, -0.13])
    plan = make_plan(
        fly, template, articulation, measurements, chain_offsets={"head": shift}
    )
    back = BodyPlan.from_json(plan.to_json(), fly)
    np.testing.assert_allclose(back.chain_offsets["head"], shift, atol=1e-12)
    assert back.plan["x-deeperfly"]["version"] == plan.plan["x-deeperfly"]["version"]


def test_a_plan_without_chain_offsets_reads_as_no_shift(
    fly, template, articulation, measurements
):
    """A version-1 plan carries no ``chain_offsets``; that has to mean zero, not missing."""
    plan = make_plan(fly, template, articulation, measurements)
    stored = json.loads(plan.to_json())
    stored["x-deeperfly"].pop("chain_offsets", None)  # as an older deeperfly wrote it
    back = BodyPlan.from_json(json.dumps(stored), fly)
    assert back.chain_offsets == {}


def test_the_overlay_mesh_and_the_plan_agree_on_the_shifted_pivot(
    fly, template, articulation, measurements
):
    """The mesh node affine lands a marker exactly where the plan's FK does.

    ``chain_scales`` already had to hold this property -- fit and overlay must describe
    one pose -- and a base shift is the second thing that can break it. Shifting a
    chain's anchors *and* its attached points by ``d`` is exactly a post-translation of
    the affine, so passing ``chain_offsets`` to :meth:`NmfMesh.pose` is the whole
    correction; the second half of this test is what omitting it costs.
    """
    from deeperfly.inverse_kinematics.mesh import load_nmf_mesh

    shift = np.array([0.03, -0.02, -0.13])
    plan = make_plan(
        fly,
        template,
        articulation,
        measurements,
        chain_scales={"head": 1.24},
        chain_offsets={"head": shift},
    )
    head = articulation.chain("head")
    mesh = load_nmf_mesh()
    col = {n: i for i, n in enumerate(plan.angle_names)}
    row = list(plan.joint_names).index("l_antenna")
    neutral = head.marker_neutral[head.marker_index("l_antenna")]

    rng = np.random.default_rng(4)
    for _ in range(8):
        angles = np.zeros((1, plan.n_dofs))
        for name, value in zip(head.dof_names, rng.uniform(-0.6, 0.6, 3)):
            angles[0, col[name]] = value
        want = plan.kinematics().joint_positions(angles, None, None)[0, row]

        a, b = mesh._node_transforms(
            angles[0], list(plan.angle_names), (1.24, 1.0), {"head": shift}
        )[(0, 3)]
        np.testing.assert_allclose(a @ neutral + b, want, atol=1e-12)

        a0, b0 = mesh._node_transforms(angles[0], list(plan.angle_names), (1.24, 1.0))[
            (0, 3)
        ]
        assert np.linalg.norm((a0 @ neutral + b0) - want) == pytest.approx(
            float(np.linalg.norm(shift)), abs=1e-12
        )
