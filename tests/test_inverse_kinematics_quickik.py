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
from helpers import bent_angles, place_chain_markers, rot_z, synth_leg_pose

from deeperfly.config import Config
from deeperfly.inverse_kinematics.articulation import load_articulation
from deeperfly.inverse_kinematics.mesh import load_nmf_mesh
from deeperfly.inverse_kinematics.template import KinematicTemplate
from deeperfly.skeleton import Skeleton

pytest.importorskip("quickik", reason="needs the deeperfly[ik] extra")

from deeperfly.inverse_kinematics import (  # noqa: E402  (after importorskip)
    solve_inverse_kinematics,
)


@pytest.fixture(scope="module")
def fly() -> Skeleton:
    return Skeleton.fly()


@pytest.fixture(scope="module")
def template() -> KinematicTemplate:
    return KinematicTemplate.load("neuromechfly")


@pytest.fixture(scope="module")
def articulation():
    return load_articulation()


def _index(skeleton) -> dict[str, int]:
    return {n: i for i, n in enumerate(skeleton.point_names)}


def _place_coxae(pts, index, articulation, sim):
    """Put the six body-fixed coxae where ``sim`` maps the model's, registering the body."""
    neutral = load_nmf_mesh().kp_neutral
    for name in articulation.coxa_points:
        pts[:, index[name]] = sim[1] * (sim[0] @ neutral[index[name]]) + sim[2]


# -- recovering known angles -------------------------------------------------


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
    markers are placed for that size, and the estimator must recover it from their
    contour length so the plan is built at the right scale.

    The abdomen is posed *straight* rather than bent, because its size estimate is only
    exact at the neutral pose -- it comes from a polyline through the per-depth marker
    centroids, which shortens as a serial chain curls (the head's three DOFs share one
    pivot, so its contour is rotation-invariant and any pose will do). That is a
    property of the estimator, not of the solve; it is pinned down directly by
    ``test_estimate_chain_scale_recovers_contour_length``.
    """
    index = _index(fly)
    chain = articulation.chain(name)
    rng = np.random.default_rng(0)
    sim = (rot_z(0.4), 1.6, np.array([2.0, -1.0, 3.0]))
    truth = (
        bent_angles(chain, rng, frac=(0.4, 0.6))
        if name == "head"
        else np.zeros(len(chain.dof_names))
    )
    pts = np.full((3, fly.n_points, 3), np.nan)
    _place_coxae(pts, index, articulation, sim)
    place_chain_markers(chain, truth, sim, pts, index, size=size)

    res = solve_inverse_kinematics(
        pts, fly, template, articulation=articulation, neutral_weight=0.0
    )
    assert res.chain_scales[name] == pytest.approx(size, abs=1e-3)
    for marker in chain.marker_names:
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
    for name in ("l_antenna", "r_antenna", "l_abdomen0", "r_abdomen2"):
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
    """Dropping the claw still fits the leg from its remaining joints."""
    rng = np.random.default_rng(5)
    pts3d, _ = synth_leg_pose(template, fly, rng)
    index = _index(fly)
    pts3d[:, index["rf_claw"]] = np.nan
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
    for name in ("rh_coxa_trochanter", "rh_femur_tibia", "rh_tibia_tarsus", "rh_claw"):
        pts3d[:, index[name]] = np.nan  # only the coxa left: one observation
    res = solve_inverse_kinematics(pts3d, fly, template, articulation=articulation)
    col = {n: i for i, n in enumerate(res.angle_names)}
    rh = next(leg for leg in template.legs if leg.name == "rh")
    assert np.isnan(res.angles[:, [col[n] for n in rh.dof_names]]).all()
    # its distal model points go with it, while another leg is unaffected
    assert np.isnan(res.model_pts3d[:, index["rh_claw"]]).all()
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
    """Too few coxae to register the body is an explanatory error, not a wrong fit."""
    pts3d = np.full((2, fly.n_points, 3), np.nan)
    index = _index(fly)
    pts3d[:, index["rf_thorax_coxa"]] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="three of the six thorax-coxa"):
        solve_inverse_kinematics(pts3d, fly, template)


# -- weights -----------------------------------------------------------------


def test_zero_weight_marks_an_observation_missing(template, fly, articulation):
    """A zero-weight keypoint is ignored exactly as a NaN one is."""
    rng = np.random.default_rng(8)
    pts3d, _ = synth_leg_pose(template, fly, rng)
    index = _index(fly)
    dropped = ("rh_coxa_trochanter", "rh_femur_tibia", "rh_tibia_tarsus", "rh_claw")

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


def test_stage_ik_pins_constant_points(fly):
    """``constant_points`` collapses a jittering keypoint to its median before the fit.

    With a free body the pinned coxa no longer drags the fitted root around frame to
    frame, so the fitted leg root is markedly steadier. (Under the default
    ``fixed_body`` the leg roots are already held at their measured medians by
    construction, so there is nothing left for this to change.)
    """
    from deeperfly.pipeline.stages import stage_inverse_kinematics

    template = KinematicTemplate.load("neuromechfly")
    rng = np.random.default_rng(12)
    pts3d, _ = synth_leg_pose(template, fly, rng, n_frames=8)
    index = _index(fly)
    coxa = index["rf_thorax_coxa"]
    pts3d[:, coxa] += rng.normal(scale=0.05, size=(8, 3))

    common = {"fit_head": False, "fit_abdomen": False, "fixed_body": False}
    off = stage_inverse_kinematics(
        Config.from_dict({"inverse_kinematics": common}), fly, pts3d.copy()
    )
    on = stage_inverse_kinematics(
        Config.from_dict(
            {"inverse_kinematics": {**common, "constant_points": ["rf_thorax_coxa"]}}
        ),
        fly,
        pts3d.copy(),
    )
    wobble = lambda r: np.ptp(r.model_pts3d[:, coxa], axis=0).max()  # noqa: E731
    assert wobble(on) < wobble(off)


def test_stage_ik_fixed_body_holds_the_leg_roots_at_their_median(fly):
    """Under ``fixed_body`` a leg root is constant over the recording, at its median.

    The physical claim for a tethered fly, and what makes the leg-root pinning
    ``constant_points`` was invented for automatic.
    """
    from deeperfly.pipeline.stages import stage_inverse_kinematics

    template = KinematicTemplate.load("neuromechfly")
    rng = np.random.default_rng(13)
    pts3d, _ = synth_leg_pose(template, fly, rng, n_frames=8)
    index = _index(fly)
    coxa = index["rf_thorax_coxa"]
    pts3d[:, coxa] += rng.normal(scale=0.05, size=(8, 3))

    res = stage_inverse_kinematics(
        Config.from_dict(
            {"inverse_kinematics": {"fit_head": False, "fit_abdomen": False}}
        ),
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
    cfg = {"fit_head": False, "fit_abdomen": False, "weigh_by_confidence": True}
    stages.stage_inverse_kinematics(
        Config.from_dict({"inverse_kinematics": cfg}), fly, pts3d, conf
    )
    np.testing.assert_allclose(seen["weights"], 0.5)
