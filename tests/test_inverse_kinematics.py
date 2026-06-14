"""Tests for the inverse-kinematics stage: FK/IK math, alignment, template, solve."""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.inverse_kinematics import solve_inverse_kinematics
from deeperfly.inverse_kinematics.align import (
    Alignment,
    body_alignment,
    to_local,
    to_world,
)
from deeperfly.inverse_kinematics.core import solve_leg
from deeperfly.inverse_kinematics.kinematics import make_leg_fk
from deeperfly.inverse_kinematics.template import KinematicTemplate
from deeperfly.skeleton import Skeleton

# Front-leg segment lengths (NMF units); shared by all legs in the synthetic pose.
_SEGLENS = np.array([0.0, 0.40, 0.69, 0.54, 0.63])

# Plausible coxa positions for the six legs in a body-aligned world frame.
_COXAE = {
    "lf": [1.0, 0.5, 0.0],
    "rf": [1.0, -0.5, 0.0],
    "lm": [0.0, 0.6, 0.0],
    "rm": [0.0, -0.6, 0.0],
    "lh": [-1.0, 0.5, 0.0],
    "rh": [-1.0, -0.5, 0.0],
}


@pytest.fixture
def template() -> KinematicTemplate:
    return KinematicTemplate.load("neuromechfly")


@pytest.fixture
def fly() -> Skeleton:
    return Skeleton.fly()


def _bent_angles(chain, rng, frac=(0.3, 0.7)):
    """Random joint angles within ``frac`` of each DOF's bounds (a non-singular pose)."""
    lo, hi = chain.bounds
    return lo + (hi - lo) * rng.uniform(frac[0], frac[1], size=len(lo))


def _synth_pose(template, fly, rng, r_body=None, n_frames=3):
    """A synthetic 3D pose: each leg placed by FK from random angles in a known frame.

    Returns ``(pts3d (T, P, 3), truth_angles {leg: (D,)})``.
    """
    r_body = np.eye(3) if r_body is None else r_body
    index = {n: i for i, n in enumerate(fly.point_names)}
    pts3d = np.full((n_frames, fly.n_points, 3), np.nan)
    truth = {}
    for leg in template.legs:
        ang = _bent_angles(leg, rng)
        truth[leg.name] = ang
        local = np.asarray(make_leg_fk(leg.dof_counts)(ang, leg.axes, _SEGLENS))
        world = to_world(local, np.array(_COXAE[leg.name]), r_body)
        for j, name in enumerate(leg.point_names):
            pts3d[:, index[name]] = world[j]
    return pts3d, truth


# -- forward / inverse kinematics --------------------------------------------


def test_fk_rest_pose_points_straight_down(template):
    """With every angle zero the chain is a straight leg along -z (the rest pose)."""
    leg = template.legs[0]
    fk = make_leg_fk(leg.dof_counts)
    joints = np.asarray(fk(np.zeros(sum(leg.dof_counts)), leg.axes, _SEGLENS))
    # x and y stay at the origin; z descends by the cumulative segment length.
    np.testing.assert_allclose(joints[:, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(joints[:, 2], -np.cumsum(_SEGLENS), atol=1e-12)


def test_fk_jacobian_is_finite_at_zero(template):
    """The FK Jacobian is finite even at angle 0 (the fixed-axis Rodrigues form)."""
    import jax

    leg = template.legs[0]
    fk = make_leg_fk(leg.dof_counts)
    jac = jax.jacrev(fk, argnums=0)(np.zeros(sum(leg.dof_counts)), leg.axes, _SEGLENS)
    assert np.isfinite(np.asarray(jac)).all()


def test_solve_leg_recovers_positions_and_angles(template, fly, rng):
    """A per-leg solve reproduces the measured joint positions (and angles) exactly."""
    leg = next(leg for leg in template.legs if leg.name == "rf")
    true = _bent_angles(leg, rng)
    local = np.asarray(make_leg_fk(leg.dof_counts)(true, leg.axes, _SEGLENS))
    origin = np.array([12.0, -3.0, 5.0])
    world = to_world(local, origin, np.eye(3))

    index = {n: i for i, n in enumerate(fly.point_names)}
    pts3d = np.full((2, fly.n_points, 3), np.nan)
    for j, name in enumerate(leg.point_names):
        pts3d[:, index[name]] = world[j]
    align = Alignment(np.eye(3), {"rf": origin}, {"rf": _SEGLENS}, None)

    angles, model = solve_leg(pts3d, index, leg, align, max_nfev=200)
    np.testing.assert_allclose(model[0], world, atol=1e-6)
    np.testing.assert_allclose(angles[0], true, atol=1e-5)


def test_fitted_angles_stay_within_bounds(template, fly, rng):
    pts3d, _ = _synth_pose(template, fly, rng)
    res = solve_inverse_kinematics(pts3d, fly, template, max_nfev=200)
    col = 0
    for leg in template.legs:
        lo, hi = leg.bounds
        d = len(lo)
        ang = res.angles[:, col : col + d]
        finite = np.isfinite(ang)
        assert (ang[finite] >= (lo - 1e-6)[None].repeat(ang.shape[0], 0)[finite]).all()
        assert (ang[finite] <= (hi + 1e-6)[None].repeat(ang.shape[0], 0)[finite]).all()
        col += d


def test_missing_distal_joints_tolerated(template, fly, rng):
    """Dropping the claw/tarsus still fits the leg from the remaining joints."""
    pts3d, _ = _synth_pose(template, fly, rng)
    index = {n: i for i, n in enumerate(fly.point_names)}
    pts3d[:, index["rf_claw"]] = np.nan  # occluded distal joint
    res = solve_inverse_kinematics(pts3d, fly, template, max_nfev=200)
    # the observed rf joints are still reproduced by the fitted model (dropping the
    # claw frees the TiTa DOF, so the proximal joints fit to a looser tolerance)
    for name in ("rf_thorax_coxa", "rf_femur_tibia", "rf_tibia_tarsus"):
        p = index[name]
        assert np.isfinite(res.model_pts3d[0, p]).all()
        np.testing.assert_allclose(res.model_pts3d[0, p], pts3d[0, p], atol=5e-3)


def test_full_solve_shapes_and_overlay_fit(template, fly, rng):
    pts3d, _ = _synth_pose(template, fly, rng, n_frames=4)
    res = solve_inverse_kinematics(pts3d, fly, template, max_nfev=200)
    # angle columns = leg DOFs (+ antenna angles appended by the head solve)
    assert res.angles.shape == (4, len(res.angle_names))
    assert res.angle_names[: len(template.dof_names)] == template.dof_names
    assert res.model_pts3d.shape == (4, fly.n_points, 3)
    # every observed leg joint is reproduced (the overlay lands on the keypoints)
    mask = np.isfinite(pts3d) & np.isfinite(res.model_pts3d)
    mask = mask.all(axis=-1)
    err = np.linalg.norm(
        np.where(mask[..., None], res.model_pts3d - pts3d, 0.0), axis=-1
    )
    assert np.max(err[mask]) < 1e-4


# -- alignment ---------------------------------------------------------------


def test_alignment_frame_is_orthonormal_and_invariant(template, fly, rng):
    """A rotated body still fits exactly; the recovered body frame is orthonormal."""
    angle = 0.5
    r = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1.0],
        ]
    )
    pts3d, _ = _synth_pose(template, fly, rng, r_body=r)
    align = body_alignment(pts3d, fly, template)
    rb = align.r_body
    np.testing.assert_allclose(rb.T @ rb, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(rb), 1.0, atol=1e-6)
    # the solve still reproduces the (rotated) measured joints
    res = solve_inverse_kinematics(pts3d, fly, template, max_nfev=200)
    p = {n: i for i, n in enumerate(fly.point_names)}["rf_claw"]
    np.testing.assert_allclose(res.model_pts3d[0, p], pts3d[0, p], atol=1e-3)


def test_to_local_to_world_round_trip(rng):
    r = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    origin = rng.normal(size=3)
    pts = rng.normal(size=(5, 3))
    back = to_world(to_local(pts, origin, r), origin, r)
    np.testing.assert_allclose(back, pts, atol=1e-10)


def test_measured_seglens_recovered(template, fly, rng):
    pts3d, _ = _synth_pose(template, fly, rng)
    align = body_alignment(pts3d, fly, template)
    np.testing.assert_allclose(align.seglens["rf"], _SEGLENS, atol=1e-6)


# -- head / antenna ----------------------------------------------------------


def test_head_antenna_angles_present_and_absent(template, fly, rng):
    pts3d, _ = _synth_pose(template, fly, rng)
    index = {n: i for i, n in enumerate(fly.point_names)}
    # place antennae forward + up of the head origin
    for name in ("l_antenna", "r_antenna"):
        pts3d[:, index[name]] = [2.0, 0.0, 0.5]
    res = solve_inverse_kinematics(pts3d, fly, template, max_nfev=120)
    assert any(n.endswith("antenna_pitch") for n in res.angle_names)
    pitch = res.angle_names.index("Angle_L_antenna_pitch")
    assert np.isfinite(res.angles[:, pitch]).all()


# -- template ----------------------------------------------------------------


def test_template_bounds_override_and_leg_restriction():
    t = KinematicTemplate.load(
        "neuromechfly", legs=["rf", "lf"], bounds_overrides={"RF_FTi_pitch": (10, 160)}
    )
    assert [leg.name for leg in t.legs] == ["rf", "lf"]
    rf = next(leg for leg in t.legs if leg.name == "rf")
    fti = next(j for j in rf.joints if j.name == "FTi").dofs[0]
    assert round(np.rad2deg(fti.lo)) == 10 and round(np.rad2deg(fti.hi)) == 160


def test_template_unknown_leg_rejected():
    with pytest.raises(ValueError, match="unknown leg"):
        KinematicTemplate.load("neuromechfly", legs=["xx"])


def test_template_unknown_name_rejected():
    with pytest.raises(FileNotFoundError):
        KinematicTemplate.load("not-a-template")


# -- NeuroMechFly mesh overlay -----------------------------------------------


def test_nmf_mesh_pose_at_neutral_is_identity():
    """Posing at the model's own neutral keypoints reproduces the baked mesh."""
    from deeperfly.inverse_kinematics.mesh import load_nmf_mesh

    mesh = load_nmf_mesh()
    verts, valid = mesh.pose(mesh.kp_neutral)
    assert valid.all()
    np.testing.assert_allclose(verts, mesh.vertices, atol=1e-3)


def test_nmf_mesh_drops_occluded_segment():
    """A leg segment with missing endpoints is dropped; the rest still poses."""
    from deeperfly.inverse_kinematics.mesh import load_nmf_mesh

    mesh = load_nmf_mesh()
    pts = mesh.kp_neutral.copy()
    pts[mesh.slot_prox[1]] = np.nan  # NaN one leg bone's endpoints
    pts[mesh.slot_dist[1]] = np.nan
    verts, valid = mesh.pose(pts)
    assert valid.any() and not valid.all()
    # the dropped bone's vertices are NaN; the body slot (0) stays finite
    body = mesh.vert_slot == 0
    assert np.isfinite(verts[body]).all()


def test_nmf_mesh_follows_a_translated_pose():
    """Shifting every keypoint shifts the whole posed mesh by the same offset."""
    from deeperfly.inverse_kinematics.mesh import load_nmf_mesh

    mesh = load_nmf_mesh()
    shift = np.array([3.0, -2.0, 1.0])
    verts, _ = mesh.pose(mesh.kp_neutral + shift)
    np.testing.assert_allclose(verts, mesh.vertices + shift, atol=1e-3)
