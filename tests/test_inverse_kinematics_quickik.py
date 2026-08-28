"""Tests for the QuickIK solve itself.

Split out from ``test_inverse_kinematics.py`` because everything here needs the optional
``deeperfly[ik]`` extra, which has no wheels and so is not installed by default (nor in
the default CI job). The model geometry, the body plan, the forward kinematics and the
overlay are all tested without it, so a skip here never hides a hole in the code a plain
install actually runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import (
    bent_angles,
    deepfly3d_skeleton,  # noqa: F401
    fly38_skeleton,
    place_chain_markers,
    rot_z,
    synth_leg_pose,
)

from deeperfly.config import Config
from deeperfly.inverse_kinematics.articulation import load_articulation
from deeperfly.inverse_kinematics.template import KinematicTemplate
from deeperfly.skeleton import Skeleton

pytest.importorskip("quickik", reason="needs the deeperfly[ik] extra")

from deeperfly.inverse_kinematics import (  # noqa: E402  (after importorskip)
    solve_inverse_kinematics,
)


@pytest.fixture(scope="module")
def fly() -> Skeleton:
    """``fly38b`` -- the skeleton both baked chains are targeted at.

    It is the one that labels the head's ``neck`` base and the abdomen's midline
    ``abdomen0..4``, so it is the only skeleton on which a body plan covers every point
    and both chains are fit. (``fly38`` keeps a home in ``test_ik_baseline.py``, whose
    recorded pose is in its order.)
    """
    return fly38_skeleton()


@pytest.fixture(scope="module")
def template() -> KinematicTemplate:
    return KinematicTemplate.load("neuromechfly")


@pytest.fixture(scope="module")
def articulation():
    return load_articulation()


def _index(skeleton) -> dict[str, int]:
    return {n: i for i, n in enumerate(skeleton.point_names)}


def _place_coxae(pts, index, articulation, sim):
    """Put the six body-fixed coxae where ``sim`` maps the model's, registering the body.

    From ``articulation.anchor_neutral``, which is paired with ``anchors`` by position,
    rather than from the mesh asset's ``kp_neutral`` indexed by the run's skeleton: that
    asset carries its own point order, and indexing it with another skeleton's silently
    reads a different point. It is also exactly the array ``_anchor_similarity`` fits
    against, so the registration this synthesizes is the one the stage recovers.
    """
    for name, neutral in zip(articulation.anchor_points, articulation.anchor_neutral):
        pts[:, index[name]] = sim[1] * (sim[0] @ np.asarray(neutral)) + sim[2]


# -- recovering known angles -------------------------------------------------


def test_the_leg_parameterisation_is_flygyms(template, fly):
    """Fitting the MODEL's own neutral pose returns the model's own joint angles.

    This is the test that pins the whole leg frame convention, and it is worth one: the
    template's DOF axes, their order, the per-side mirroring and the plan's ``offset_quat``
    all have to agree with the MJCF at once, and every way of getting one of them wrong
    still produces a plausible-looking fly. Three of them *were* wrong, and nothing caught
    it, because a leg fitted in the wrong frame still lands near the keypoints -- it just
    reports angles that are not flygym's, and needs joint limits that are not
    NeuroMechFly's to reach the pose.

    The check needs no MuJoCo and no synthesis. the pack's ``mesh.npz`` carries the model's
    neutral keypoint positions and the articulation asset carries its spring references,
    so: feed the former in as a recording, and the fitted angles must come back as the
    latter. If they do, deeperfly's angles ARE flygym's angles and its limits mean what
    NeuroMechFly says.

    The tolerance is 8 degrees, and one DOF needs it: the tibia-tarsus pitch, because
    deeperfly models the tarsus as ONE straight segment from ``tibia_tarsus`` to ``pretarsus``
    where flygym has five tarsal joints, so the single angle splits the difference. Every
    other DOF lands within 5 degrees and the median is under 1.
    """
    from deeperfly.inverse_kinematics.mesh import load_model_mesh

    mesh = load_model_mesh()
    kp = np.asarray(mesh.kp_neutral, dtype=float)
    assert kp.shape[0] == fly.n_points, "the mesh asset is not this skeleton's"
    rest = load_articulation().leg_rest
    assert rest, "the articulation asset carries no leg spring references"

    result = solve_inverse_kinematics(np.repeat(kp[None], 3, axis=0), fly, template)
    col = {n: i for i, n in enumerate(result.angle_names)}
    errs = {
        name: abs((result.angles[0, col[name]] - want + np.pi) % (2 * np.pi) - np.pi)
        for name, want in rest.items()
    }
    worst = max(errs, key=errs.get)
    assert np.rad2deg(errs[worst]) < 8.0, (
        f"{worst} is {np.rad2deg(errs[worst]):.1f} deg from the model's rest angle"
    )
    assert np.rad2deg(np.median(list(errs.values()))) < 1.0
    assert sum(np.rad2deg(e) < 5.0 for e in errs.values()) >= len(errs) - 4

    # and the pose it reaches really is the model's, not merely a self-consistent one
    fitted = result.model_pts3d[0]
    seen = np.isfinite(fitted).all(axis=-1)
    assert np.nanmax(np.linalg.norm(fitted[seen] - kp[seen], axis=-1)) < 0.02


def test_recovers_the_generating_leg_angles(template, fly, articulation):
    """A pose built by forward kinematics from known angles solves back to them.

    Ground truth that outlives any particular solver: the pose is exactly reachable, so
    a correct solve must reproduce the keypoints. The *angles* are checked loosely --
    a leg's thorax-coxa has three DOFs but its next joint's position gives only two
    constraints, so the chain is genuinely redundant and several angle triples describe
    the same keypoints. The keypoints are the contract; the angle check is a sanity
    bound on how far into that null space the solve wanders.
    """
    rng = np.random.default_rng(20260729)
    pts3d, truth = synth_leg_pose(template, fly, rng, n_frames=3)
    res = solve_inverse_kinematics(
        pts3d, fly, template, articulation=articulation, neutral_weight=1e-5
    )
    index = _index(fly)
    leg_points = [index[p] for leg in template.legs for p in leg.point_names]
    err = np.linalg.norm(res.model_pts3d[:, leg_points] - pts3d[:, leg_points], axis=-1)
    # ~4% of the shortest bone. The floor here is the damping, which is set high
    # enough to keep the ill-conditioned chains out of their limits (see
    # InverseKinematicsParams.damping) and biases the fixed point a little in return.
    assert np.nanmax(err) < 2e-2

    col = {n: i for i, n in enumerate(res.angle_names)}
    for leg in template.legs:
        got = np.array([res.angles[0, col[n]] for n in leg.dof_names])
        assert np.degrees(np.abs(got - truth[leg.name])).max() < 25.0


@pytest.mark.parametrize(
    "name,size", [("head", 1.0), ("head", 1.6), ("abdomen", 1.0), ("abdomen", 0.7)]
)
def test_recovers_the_generating_chain_angles(template, fly, articulation, name, size):
    """A head/abdomen chain solves back to the angles that placed its markers.

    ``size`` exercises a chain grown or shrunk relative to the model geometry: the
    markers are placed for that size, and the estimator must recover it so the plan is
    built at the right scale.

    Both chains are posed **bent**, which is the whole point on the abdomen: its markers
    sit on the dorsal surface, the outside of a ventral bend, so a ruler drawn along the
    chain would read a curled abdomen as a much larger one. The estimator only compares
    separations the chain's own joints cannot change, so posture cannot leak into size --
    pinned directly by ``test_estimate_chain_scale_recovers_a_resized_chain_at_a_bent_pose``.

    The head is synthesized at a base the model does not share, which is the real case
    since the coxa registration cannot locate the head pivot. Both the shift and the size
    have to come back, and they are not separable by the solve: read the size from the
    model's own anchor and a displaced base reads as a bigger head, which then needs a
    rotation to reach the antennae.
    """
    index = _index(fly)
    chain = articulation.chain(name)
    rng = np.random.default_rng(0)
    sim = (rot_z(0.4), 1.6, np.array([2.0, -1.0, 3.0]))
    truth = bent_angles(chain, rng, frac=(0.4, 0.6))
    shift = np.array([0.04, -0.03, -0.14]) if name == "head" else None
    pts = np.full((3, fly.n_points, 3), np.nan)
    _place_coxae(pts, index, articulation, sim)
    place_chain_markers(chain, truth, sim, pts, index, size=size, shift=shift)

    res = solve_inverse_kinematics(
        pts, fly, template, articulation=articulation, neutral_weight=0.0
    )
    assert res.chain_scales[name] == pytest.approx(size, abs=1e-3)
    if shift is not None:
        np.testing.assert_allclose(res.chain_offsets[name], shift, atol=1e-6)
    for marker in chain.marker_names:
        if marker not in index:
            continue  # a model marker this skeleton does not label
        np.testing.assert_allclose(
            res.model_pts3d[0, index[marker]], pts[0, index[marker]], atol=5e-3
        )
    col = {n: i for i, n in enumerate(res.angle_names)}
    got = np.array([res.angles[0, col[n]] for n in chain.dof_names])
    assert np.degrees(np.abs(got - truth)).max() < 5.0


def test_a_straight_abdomen_does_not_curl(template, fly, articulation):
    """A straight abdomen stays straight -- the clamping-deadlock regression test.

    The abdomen's five near-collinear hinges make its Jacobian ill-conditioned, and its
    limits are ``[-30, 0]`` degrees with the straight pose sitting exactly on the upper
    bound. With light Levenberg-Marquardt damping the first Gauss-Newton step overshoots
    past the lower bound; QuickIK clamps it there and, having no gradient projection,
    never recovers -- an exactly-straight abdomen converges to a *full ventral curl* and
    stays there however many iterations it is given. This asserts the default damping is
    heavy enough to prevent that.
    """
    index = _index(fly)
    chain = articulation.chain("abdomen")
    sim = (rot_z(0.4), 1.6, np.array([2.0, -1.0, 3.0]))
    pts = np.full((2, fly.n_points, 3), np.nan)
    _place_coxae(pts, index, articulation, sim)
    place_chain_markers(chain, np.zeros(len(chain.dof_names)), sim, pts, index)

    res = solve_inverse_kinematics(pts, fly, template, articulation=articulation)
    col = {n: i for i, n in enumerate(res.angle_names)}
    got = res.angles[0, [col[n] for n in chain.dof_names]]
    assert np.degrees(np.abs(got)).max() < 2.0  # was a full -30 deg at every hinge


def test_solve_shapes_and_angle_names(template, fly, articulation):
    """The output contract: shapes, and leg DOFs first in the documented name scheme."""
    rng = np.random.default_rng(1)
    pts3d, _ = synth_leg_pose(template, fly, rng, n_frames=4)
    res = solve_inverse_kinematics(pts3d, fly, template, articulation=articulation)
    assert res.angles.shape == (4, len(res.angle_names))
    assert res.angle_names[: len(template.dof_names)] == template.dof_names
    assert res.model_pts3d.shape == (4, fly.n_points, 3)
    assert "c_thorax-c_head-yaw" in res.angle_names
    assert "c_abdomen12-c_abdomen3-pitch" in res.angle_names
    assert res.body_plan is not None
    assert list(res.body_plan.angle_names) == res.angle_names


def test_solve_fills_head_and_abdomen_markers(template, fly, articulation):
    """With an articulation the antenna and abdomen model points are filled in."""
    index = _index(fly)
    sim = (rot_z(0.2), 1.5, np.array([1.0, 2.0, -1.0]))
    rng = np.random.default_rng(3)
    pts = np.full((2, fly.n_points, 3), np.nan)
    _place_coxae(pts, index, articulation, sim)
    for chain in articulation.chains:
        place_chain_markers(
            chain, bent_angles(chain, rng, frac=(0.4, 0.6)), sim, pts, index
        )
    res = solve_inverse_kinematics(pts, fly, template, articulation=articulation)
    for name in ("l_antenna", "r_antenna", "neck", "abdomen0", "abdomen4"):
        assert np.isfinite(res.model_pts3d[0, index[name]]).all()


# -- joint limits ------------------------------------------------------------


def test_fitted_angles_stay_within_bounds(template, fly, articulation):
    """Every solved angle respects its joint limits.

    QuickIK enforces limits by clamping each Gauss-Newton step rather than with a
    bounded trust region, so this is the only thing pinning down that the box is
    actually honoured.
    """
    rng = np.random.default_rng(2)
    pts3d, _ = synth_leg_pose(template, fly, rng)
    res = solve_inverse_kinematics(pts3d, fly, template, articulation=articulation)
    col = {n: i for i, n in enumerate(res.angle_names)}
    for chain in (*template.legs, *articulation.chains):
        lo, hi = chain.bounds
        names = chain.dof_names
        got = res.angles[:, [col[n] for n in names]]
        finite = np.isfinite(got)
        assert (got[finite] >= np.broadcast_to(lo, got.shape)[finite] - 1e-6).all()
        assert (got[finite] <= np.broadcast_to(hi, got.shape)[finite] + 1e-6).all()


def test_pinned_joint_limits_are_reported(template, fly, articulation, caplog):
    """A limit that is capping the fit is logged, not left as an unexplained residual.

    With the box clamped shut on a DOF, the limit -- not the data -- decides the angle,
    and the user's lever is ``[inverse_kinematics.bounds]``. Worth saying so: the
    packaged template's middle and hind legs reuse the *front* leg's ranges as a
    placeholder, which is exactly the case where this bites.
    """
    rng = np.random.default_rng(4)
    pts3d, _ = synth_leg_pose(template, fly, rng)
    shut = KinematicTemplate.load(
        "neuromechfly",
        bounds_overrides={"c_thorax-rf_coxa-yaw": (40.0, 41.0)},
    )
    with caplog.at_level("WARNING", logger="deeperfly"):
        solve_inverse_kinematics(pts3d, fly, shut, articulation=articulation)
    assert "sit at a limit in most frames" in caplog.text
    assert "c_thorax-rf_coxa-yaw" in caplog.text


# -- missing observations ----------------------------------------------------


def test_missing_distal_joint_still_fits_the_rest(template, fly, articulation):
    """Dropping the pretarsus still fits the leg from its remaining joints."""
    rng = np.random.default_rng(5)
    pts3d, _ = synth_leg_pose(template, fly, rng)
    index = _index(fly)
    pts3d[:, index["rf_pretarsus"]] = np.nan
    res = solve_inverse_kinematics(
        pts3d, fly, template, articulation=articulation, neutral_weight=1e-5
    )
    for name in ("rf_thorax_coxa", "rf_femur_tibia", "rf_tibia_tarsus"):
        p = index[name]
        assert np.isfinite(res.model_pts3d[0, p]).all()
        np.testing.assert_allclose(res.model_pts3d[0, p], pts3d[0, p], atol=1e-2)


def test_a_limb_without_enough_observations_is_left_unset(template, fly, articulation):
    """A limb with fewer than two observed keypoints gets NaN angles, not a guess.

    QuickIK returns *an* angle for every DOF no matter how little data there is (pulled
    toward neutral), so without an explicit mask the overlay would confidently draw a
    limb that was never observed, and the "tracks solved" count would read 100% forever.
    """
    rng = np.random.default_rng(6)
    pts3d, _ = synth_leg_pose(template, fly, rng)
    index = _index(fly)
    for name in (
        "rh_coxa_trochanter",
        "rh_femur_tibia",
        "rh_tibia_tarsus",
        "rh_pretarsus",
    ):
        pts3d[:, index[name]] = np.nan  # only the coxa left: one observation
    res = solve_inverse_kinematics(pts3d, fly, template, articulation=articulation)
    col = {n: i for i, n in enumerate(res.angle_names)}
    rh = next(leg for leg in template.legs if leg.name == "rh")
    assert np.isnan(res.angles[:, [col[n] for n in rh.dof_names]]).all()
    # its distal model points go with it, while another leg is unaffected
    assert np.isnan(res.model_pts3d[:, index["rh_pretarsus"]]).all()
    lf = next(leg for leg in template.legs if leg.name == "lf")
    assert np.isfinite(res.angles[:, [col[n] for n in lf.dof_names]]).all()


def test_a_leg_with_no_coxa_at_all_is_left_unset(template, fly, articulation, caplog):
    """A leg whose root was never triangulated cannot be placed, and says so."""
    rng = np.random.default_rng(7)
    pts3d, _ = synth_leg_pose(template, fly, rng)
    index = _index(fly)
    for leg in template.legs:
        if leg.name == "lm":
            for point in leg.point_names:
                pts3d[:, index[point]] = np.nan
    with caplog.at_level("WARNING", logger="deeperfly"):
        res = solve_inverse_kinematics(pts3d, fly, template, articulation=articulation)
    col = {n: i for i, n in enumerate(res.angle_names)}
    lm = next(leg for leg in template.legs if leg.name == "lm")
    assert np.isnan(res.angles[:, [col[n] for n in lm.dof_names]]).all()
    assert "no observed thorax-coxa" in caplog.text


def test_registration_failure_is_a_clear_error(template, fly):
    """Too few anchors to register the body is an explanatory error, not a wrong fit."""
    pts3d = np.full((2, fly.n_points, 3), np.nan)
    index = _index(fly)
    pts3d[:, index["rf_thorax_coxa"]] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="three of its 6 anchor points"):
        solve_inverse_kinematics(pts3d, fly, template)


# -- weights -----------------------------------------------------------------


def test_zero_weight_marks_an_observation_missing(template, fly, articulation):
    """A zero-weight keypoint is ignored exactly as a NaN one is."""
    rng = np.random.default_rng(8)
    pts3d, _ = synth_leg_pose(template, fly, rng)
    index = _index(fly)
    dropped = (
        "rh_coxa_trochanter",
        "rh_femur_tibia",
        "rh_tibia_tarsus",
        "rh_pretarsus",
    )

    weights = np.ones((pts3d.shape[0], fly.n_points))
    for name in dropped:
        weights[:, index[name]] = 0.0
    by_weight = solve_inverse_kinematics(
        pts3d, fly, template, articulation=articulation, weights=weights
    )
    nan_pose = pts3d.copy()
    for name in dropped:
        nan_pose[:, index[name]] = np.nan
    by_nan = solve_inverse_kinematics(
        nan_pose, fly, template, articulation=articulation
    )
    col = {n: i for i, n in enumerate(by_weight.angle_names)}
    rh = next(leg for leg in template.legs if leg.name == "rh")
    cols = [col[n] for n in rh.dof_names]
    assert np.isnan(by_weight.angles[:, cols]).all()
    assert np.isnan(by_nan.angles[:, cols]).all()


# -- reproducibility ---------------------------------------------------------


def test_solve_is_bit_reproducible(template, fly, articulation):
    """The same input solves to exactly the same angles twice.

    Non-negotiable for a scientific pipeline, and worth pinning: the sequential solver
    warm-starts frame to frame, so any leaked state between runs would show up here.
    """
    rng = np.random.default_rng(9)
    pts3d, _ = synth_leg_pose(template, fly, rng, n_frames=5)
    kw = dict(articulation=articulation)
    a = solve_inverse_kinematics(pts3d, fly, template, **kw)
    b = solve_inverse_kinematics(pts3d, fly, template, **kw)
    np.testing.assert_array_equal(a.angles, b.angles)
    np.testing.assert_array_equal(a.model_pts3d, b.model_pts3d)


def test_parallel_solve_runs_and_stays_within_bounds(template, fly, articulation):
    """The segmented-parallel path solves the same sequence, limits still honoured.

    It is off by default because each segment restarts from the neutral pose, so the
    traces can step at a seam -- but it must still produce a valid, in-bounds fit.
    """
    rng = np.random.default_rng(10)
    pts3d, _ = synth_leg_pose(template, fly, rng, n_frames=40)
    res = solve_inverse_kinematics(
        pts3d,
        fly,
        template,
        articulation=articulation,
        parallel=True,
        segment_len=12,
        overlap_len=3,
    )
    assert res.angles.shape[0] == 40
    col = {n: i for i, n in enumerate(res.angle_names)}
    for leg in template.legs:
        lo, hi = leg.bounds
        got = res.angles[:, [col[n] for n in leg.dof_names]]
        finite = np.isfinite(got)
        assert (got[finite] >= np.broadcast_to(lo, got.shape)[finite] - 1e-6).all()
        assert (got[finite] <= np.broadcast_to(hi, got.shape)[finite] + 1e-6).all()


# -- registration ------------------------------------------------------------


def test_body_scale_comes_from_the_one_registration(template, fly, articulation):
    """``body_scale`` is the coxa registration's scale -- the same one placing the plan.

    Previously the overlay's body scale was fit a second time, from the mesh asset's own
    copy of the neutral coxae; deriving both from one registration removes the chance of
    the fit and the overlay disagreeing.
    """
    index = _index(fly)
    sim = (rot_z(0.3), 1.37, np.array([0.5, -0.25, 2.0]))
    pts = np.full((3, fly.n_points, 3), np.nan)
    _place_coxae(pts, index, articulation, sim)
    rng = np.random.default_rng(11)
    for chain in articulation.chains:
        place_chain_markers(chain, bent_angles(chain, rng), sim, pts, index)
    res = solve_inverse_kinematics(pts, fly, template, articulation=articulation)
    assert res.body_scale == pytest.approx(1.37, rel=1e-3)
    assert res.body_plan is not None
    assert res.body_plan.body_sim[1] == pytest.approx(res.body_scale)


# -- the stage ---------------------------------------------------------------


def test_stage_ik_fixed_body_holds_the_leg_roots_at_their_median(fly):
    """Under ``fixed_body`` a leg root is constant over the recording, at its median.

    The physical claim for a tethered fly, and what makes any separate leg-root pin
    unnecessary: the body plan registers from the median coxae by construction.
    """
    from deeperfly.pipeline.stages import stage_inverse_kinematics

    template = KinematicTemplate.load("neuromechfly")
    rng = np.random.default_rng(13)
    pts3d, _ = synth_leg_pose(template, fly, rng, n_frames=8)
    index = _index(fly)
    coxa = index["rf_thorax_coxa"]
    pts3d[:, coxa] += rng.normal(scale=0.05, size=(8, 3))

    res = stage_inverse_kinematics(
        Config.from_dict({"inverse_kinematics": {"chains": []}}),
        fly,
        pts3d,
    )
    assert np.ptp(res.model_pts3d[:, coxa], axis=0).max() < 1e-9
    np.testing.assert_allclose(
        res.model_pts3d[0, coxa], np.nanmedian(pts3d[:, coxa], axis=0), atol=1e-6
    )


def test_stage_ik_weighs_by_confidence_when_asked(fly, monkeypatch):
    """``weigh_by_confidence`` forwards mean per-view confidence as solve weights."""
    from deeperfly.pipeline import stages

    template = KinematicTemplate.load("neuromechfly")
    rng = np.random.default_rng(14)
    pts3d, _ = synth_leg_pose(template, fly, rng, n_frames=3)
    conf = np.full((2, 3, fly.n_points), 0.5)

    seen = {}
    real = (
        stages.solve_inverse_kinematics
        if hasattr(stages, "solve_inverse_kinematics")
        else None
    )
    del real

    import deeperfly.inverse_kinematics as ik_mod

    original = ik_mod.solve_inverse_kinematics

    def spy(*args, **kwargs):
        seen["weights"] = kwargs.get("weights")
        return original(*args, **kwargs)

    monkeypatch.setattr(ik_mod, "solve_inverse_kinematics", spy)
    cfg = {"chains": [], "weigh_by_confidence": True}
    stages.stage_inverse_kinematics(
        Config.from_dict({"inverse_kinematics": cfg}), fly, pts3d, conf
    )
    np.testing.assert_allclose(seen["weights"], 0.5)


# -- absence: a keypoint that is not on this animal ---------------------------


def test_an_amputated_leg_still_fits_the_stump(template, fly, articulation):
    """Declaring the distal joints absent must not write off the whole leg.

    An amputated leg leaves a stump whose remaining joints are real, and measuring them is
    exactly what a leg-loss study exists to do. Two surviving markers already clear the
    flat threshold, so the declaration must not *change* that -- this pins the invariance,
    which is the property that would silently regress if absence started masking
    per-branch instead of per-DOF.
    """
    rng = np.random.default_rng(11)
    pts3d, _ = synth_leg_pose(template, fly, rng)
    index = _index(fly)
    absent = np.zeros(len(fly.point_names), dtype=bool)
    for name in ("lf_femur_tibia", "lf_tibia_tarsus", "lf_pretarsus"):
        pts3d[:, index[name]] = np.nan
        absent[index[name]] = True

    without = solve_inverse_kinematics(pts3d, fly, template, articulation=articulation)
    with_decl = solve_inverse_kinematics(
        pts3d, fly, template, articulation=articulation, absent_points=absent
    )
    col = {n: i for i, n in enumerate(with_decl.angle_names)}
    lf = next(leg for leg in template.legs if leg.name == "lf")
    cols = [col[n] for n in lf.dof_names]

    # The stump is fitted, declared or not -- and the declaration does not perturb it.
    assert np.isfinite(with_decl.angles[:, cols]).any()
    np.testing.assert_allclose(
        without.angles[:, cols], with_decl.angles[:, cols], atol=1e-9
    )
    # ... and no other leg's angles moved because of the declaration.
    other = [col[n] for leg in template.legs if leg.name != "lf" for n in leg.dof_names]
    np.testing.assert_allclose(
        without.angles[:, other], with_decl.angles[:, other], atol=1e-9
    )


def test_declaring_absence_cannot_buy_an_underdetermined_fit(
    template, fly, articulation
):
    """Relaxing the threshold stops where the data can no longer determine the branch.

    A leg amputated down to its coxa leaves one marker -- three coordinates against seven
    DOFs. Lowering the bar to "one observation" there would hand back QuickIK's
    neutral-biased answer dressed up as a measurement, so the branch stays unset.
    """
    rng = np.random.default_rng(14)
    pts3d, _ = synth_leg_pose(template, fly, rng)
    index = _index(fly)
    absent = np.zeros(len(fly.point_names), dtype=bool)
    for name in (
        "lf_coxa_trochanter",
        "lf_femur_tibia",
        "lf_tibia_tarsus",
        "lf_pretarsus",
    ):
        pts3d[:, index[name]] = np.nan
        absent[index[name]] = True

    res = solve_inverse_kinematics(
        pts3d, fly, template, articulation=articulation, absent_points=absent
    )
    col = {n: i for i, n in enumerate(res.angle_names)}
    lf = next(leg for leg in template.legs if leg.name == "lf")
    assert np.isnan(res.angles[:, [col[n] for n in lf.dof_names]]).all()


def test_both_antennae_absent_leaves_the_head_unfitted(template, articulation):
    """The ``neck`` is a landmark, not evidence -- so it cannot rescue a headless fit.

    The neck sits on the head's own rotation center, so all three head DOFs leave it
    exactly where it was. Counting it would make the head look observable with one
    marker (3 coordinates against 3 DOFs, which is the relaxation a *unilateral*
    ablation relies on), and QuickIK would return its neutral-biased answer for angles
    nothing measured. This is the case that separates "places the chain" from
    "constrains the chain".
    """
    fly = fly38_skeleton()
    index = _index(fly)
    sim = (rot_z(0.2), 1.5, np.array([1.0, 2.0, -1.0]))
    rng = np.random.default_rng(12)
    pts = np.full((2, fly.n_points, 3), np.nan)
    _place_coxae(pts, index, articulation, sim)
    for chain in articulation.chains:
        place_chain_markers(
            chain, bent_angles(chain, rng, frac=(0.4, 0.6)), sim, pts, index
        )
    absent = np.zeros(len(fly.point_names), dtype=bool)
    for name in ("l_antenna", "r_antenna"):  # a bilateral ablation
        pts[:, index[name]] = np.nan
        absent[index[name]] = True
    assert np.isfinite(pts[:, index["neck"]]).all(), "the neck is still measured"

    res = solve_inverse_kinematics(
        pts, fly, template, articulation=articulation, absent_points=absent
    )
    head = next(c for c in articulation.chains if c.name == "head")
    cols = [i for i, n in enumerate(res.angle_names) if n in set(head.dof_names)]
    assert cols, "the head chain contributes DOFs"
    assert np.isnan(res.angles[:, cols]).all()
    # ...while the neck itself is still *placed*: the plan knows where the pivot is.
    assert np.isfinite(res.model_pts3d[:, index["neck"]]).all()


def test_the_head_is_placed_on_the_measured_neck(template, articulation):
    """The head chain's base goes where the neck was measured, not where the coxae say.

    The six thorax-coxae are very nearly coplanar, so the similarity fit through them
    barely determines the dorsal direction the head pivot sits 0.4 above -- which is why
    the chain is placed on its own landmark instead, exactly as each leg is placed on its
    measured median thorax-coxa.
    """
    fly = fly38_skeleton()
    index = _index(fly)
    sim = (rot_z(0.35), 1.3, np.array([-1.0, 0.5, 2.0]))
    rng = np.random.default_rng(3)
    shift = np.array([0.03, -0.02, -0.13])
    pts = np.full((4, fly.n_points, 3), np.nan)
    _place_coxae(pts, index, articulation, sim)
    head = next(c for c in articulation.chains if c.name == "head")
    place_chain_markers(
        head, bent_angles(head, rng, frac=(0.4, 0.6)), sim, pts, index, shift=shift
    )

    res = solve_inverse_kinematics(pts, fly, template, articulation=articulation)
    np.testing.assert_allclose(res.chain_offsets["head"], shift, atol=1e-9)
    plan = res.body_plan
    root = next(j for j in plan.plan["joints"] if j["name"] == head.dof_names[0])
    np.testing.assert_allclose(
        root["offset_pos"], np.asarray(head.anchors[0]) + shift, atol=1e-9
    )
    # and the fitted neck lands on the measured one, in world coordinates
    np.testing.assert_allclose(
        res.model_pts3d[0, index["neck"]], pts[0, index["neck"]], atol=1e-6
    )


def test_one_absent_antenna_leaves_the_head_fittable(template, fly, articulation):
    """The head chain has three DOFs but exactly two markers, the antennae.

    So a flat two-joint threshold makes a routine unilateral antennal ablation NaN all
    three head DOFs in every frame, forever -- while one antenna is three coordinates
    against three DOFs, which is determinable. The threshold has to be judged against what
    the branch can still deliver.
    """
    index = _index(fly)
    sim = (rot_z(0.2), 1.5, np.array([1.0, 2.0, -1.0]))
    rng = np.random.default_rng(12)
    pts = np.full((2, fly.n_points, 3), np.nan)
    _place_coxae(pts, index, articulation, sim)
    for chain in articulation.chains:
        place_chain_markers(
            chain, bent_angles(chain, rng, frac=(0.4, 0.6)), sim, pts, index
        )
    absent = np.zeros(len(fly.point_names), dtype=bool)
    pts[:, index["l_antenna"]] = np.nan  # a unilateral antennal ablation
    absent[index["l_antenna"]] = True

    head = next(c for c in articulation.chains if c.name == "head")
    head_dofs = set(head.dof_names)

    without = solve_inverse_kinematics(pts, fly, template, articulation=articulation)
    with_decl = solve_inverse_kinematics(
        pts, fly, template, articulation=articulation, absent_points=absent
    )
    cols = [i for i, n in enumerate(with_decl.angle_names) if n in head_dofs]
    assert cols, "the head chain contributes DOFs"
    assert np.isnan(without.angles[:, cols]).all()  # undeclared: head lost entirely
    assert np.isfinite(with_decl.angles[:, cols]).all()  # declared: still fitted


def test_a_missing_detection_does_not_lower_the_bar(template, fly, articulation):
    """Only a *declaration* relaxes the threshold -- never a missing detection.

    A leg the detector lost is a tracking failure; reporting neutral-biased angles for it
    would be a fabricated measurement. This is the invariant that rules out inferring
    absence from "never observed".
    """
    rng = np.random.default_rng(13)
    pts3d, _ = synth_leg_pose(template, fly, rng)
    index = _index(fly)
    for name in (
        "rh_coxa_trochanter",
        "rh_femur_tibia",
        "rh_tibia_tarsus",
        "rh_pretarsus",
    ):
        pts3d[:, index[name]] = np.nan  # only the coxa left, nothing declared absent
    res = solve_inverse_kinematics(
        pts3d,
        fly,
        template,
        articulation=articulation,
        absent_points=np.zeros(len(fly.point_names), dtype=bool),
    )
    col = {n: i for i, n in enumerate(res.angle_names)}
    rh = next(leg for leg in template.legs if leg.name == "rh")
    assert np.isnan(res.angles[:, [col[n] for n in rh.dof_names]]).all()


def test_symmetric_segments_constrains_the_animal_not_its_pose(
    template, fly, articulation
):
    """Shared bones must not become shared angles.

    The distinction is the whole design: a fly's left and right femurs are the same
    length (a fact about the animal, and what the option imposes), while a leg's
    left/right asymmetry at any instant IS the behavior (a fact about the pose, which
    nothing here may touch). The synthetic pose gives all six legs the *same* bones and
    independent random angles, so the option has nothing to correct and the fit must be
    the un-symmetrized one to the solver's own reproducibility.
    """
    rng = np.random.default_rng(20260818)
    pts3d, _ = synth_leg_pose(template, fly, rng, n_frames=3)
    kw = dict(articulation=articulation, neutral_weight=1e-5)
    free = solve_inverse_kinematics(pts3d, fly, template, **kw)
    shared = solve_inverse_kinematics(
        pts3d, fly, template, symmetric_segments=True, **kw
    )
    np.testing.assert_allclose(shared.angles, free.angles, atol=1e-12)

    col = {n: i for i, n in enumerate(shared.angle_names)}
    by_name = {leg.name: leg for leg in template.legs}

    def dofs(leg_name):
        return shared.angles[0, [col[n] for n in by_name[leg_name].dof_names]]

    for left, right in (("lf", "rf"), ("lm", "rm"), ("lh", "rh")):
        assert np.abs(dofs(left) - dofs(right)).max() > 0.1, (
            f"{left}/{right} were driven to the same pose"
        )


def test_symmetric_segments_fits_one_animal_to_a_lopsided_measurement(
    template, fly, articulation
):
    """A pose whose left legs measure 10% longer fits with mirror-equal model bones.

    That is the case the option exists for -- the real one, where each side is
    triangulated from its own camera triplet. The *fitted model* is what has to come out
    symmetric; the observations stay as measured.
    """
    rng = np.random.default_rng(11)
    pts3d, _ = synth_leg_pose(template, fly, rng, n_frames=2)
    index = _index(fly)
    for leg in template.legs:  # stretch the left legs about their own coxae
        if leg.side != "l":
            continue
        coxa = pts3d[:, index[leg.point_names[0]]]
        for name in leg.point_names[1:]:
            pts3d[:, index[name]] = coxa + 1.10 * (pts3d[:, index[name]] - coxa)

    res = solve_inverse_kinematics(
        pts3d,
        fly,
        template,
        articulation=articulation,
        symmetric_segments=True,
        neutral_weight=1e-5,
    )

    def bones(leg_name):
        chain = next(x for x in template.legs if x.name == leg_name)
        pts = res.model_pts3d[0, [index[p] for p in chain.point_names]]
        return np.linalg.norm(np.diff(pts, axis=0), axis=-1)

    for left, right in (("lf", "rf"), ("lm", "rm"), ("lh", "rh")):
        np.testing.assert_allclose(bones(left), bones(right), rtol=2e-3)
