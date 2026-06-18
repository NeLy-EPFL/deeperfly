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
    assert any(n.endswith("pedicel-pitch") for n in res.angle_names)
    pitch = res.angle_names.index("c_head-l_pedicel-pitch")
    assert np.isfinite(res.angles[:, pitch]).all()


# -- template ----------------------------------------------------------------


def test_template_bounds_override_and_leg_restriction():
    t = KinematicTemplate.load(
        "neuromechfly",
        legs=["rf", "lf"],
        bounds_overrides={"rf_trochanterfemur-rf_tibia-pitch": (10, 160)},
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


def _terminal_leg_slots(mesh) -> list[int]:
    """Leg slots whose distal keypoint (the claw) is no other segment's proximal one."""
    prox = {int(p) for p in mesh.slot_prox if p >= 0}
    return [
        s
        for s in range(int(mesh.slot_prox.shape[0]))
        if mesh.slot_prox[s] >= 0
        and mesh.slot_dist[s] >= 0
        and int(mesh.slot_dist[s]) not in prox
    ]


def test_nmf_mesh_pose_at_neutral_is_identity():
    """Posing at the model's own neutral keypoints reproduces the baked mesh.

    The terminal leg segment (the tarsus) is the one exception: it is stretched to
    reach the claw keypoint (see :meth:`NmfMesh._dist_anchor`), so its vertices are
    excluded here and checked by :func:`test_nmf_mesh_tarsus_tip_reaches_claw`.
    """
    from deeperfly.inverse_kinematics.mesh import load_nmf_mesh

    mesh = load_nmf_mesh()
    verts, valid = mesh.pose(mesh.kp_neutral)
    assert valid.all()
    keep = ~np.isin(mesh.vert_slot, _terminal_leg_slots(mesh))
    np.testing.assert_allclose(verts[keep], mesh.vertices[keep], atol=1e-3)


def test_nmf_mesh_tarsus_tip_reaches_claw():
    """The terminal leg segment's mesh tip lands on its claw keypoint.

    The baked tarsus mesh stops ~11% short of the claw, so the model skeleton's claw
    juts past the mesh tip; skinning stretches the segment so its farthest vertex
    reaches the claw keypoint (where the skeleton draws it).
    """
    from deeperfly.inverse_kinematics.mesh import load_nmf_mesh

    mesh = load_nmf_mesh()
    verts, _ = mesh.pose(mesh.kp_neutral)
    terminal = _terminal_leg_slots(mesh)
    assert len(terminal) == 6  # one tarsus per leg
    for s in terminal:
        a0 = mesh.kp_neutral[int(mesh.slot_prox[s])]
        claw = mesh.kp_neutral[int(mesh.slot_dist[s])]
        u = (claw - a0) / np.linalg.norm(claw - a0)
        v = verts[mesh.vert_slot == s]
        reach = float(((v - a0) @ u).max() / np.linalg.norm(claw - a0))
        assert reach == pytest.approx(1.0, abs=1e-3)  # was ~0.89 (11% short)


def test_nmf_mesh_drops_occluded_segment():
    """A leg segment with missing endpoints is dropped; the rest still poses."""
    from deeperfly.inverse_kinematics.mesh import load_nmf_mesh

    mesh = load_nmf_mesh()
    pts = mesh.kp_neutral.copy()
    bone = int(np.flatnonzero(mesh.slot_prox >= 0)[0])  # first leg-bone slot
    pts[mesh.slot_prox[bone]] = np.nan  # NaN one leg bone's endpoints
    pts[mesh.slot_dist[bone]] = np.nan
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
    base, _ = mesh.pose(mesh.kp_neutral)
    verts, _ = mesh.pose(mesh.kp_neutral + shift)
    np.testing.assert_allclose(verts, base + shift, atol=1e-3)


def test_nmf_mesh_fixed_body_scale_holds_size_constant():
    """A fixed ``body_scale`` places the body at that size regardless of the coxae."""
    from deeperfly.inverse_kinematics.mesh import load_nmf_mesh

    mesh = load_nmf_mesh()
    # Spread the coxae out by 1.5x about their centroid: a per-frame fit would scale
    # the body up ~1.5x, but a fixed body_scale must keep the rigid body's size.
    coxae = mesh.kp_neutral[mesh.coxa_idx]
    pts = mesh.kp_neutral.copy()
    pts[mesh.coxa_idx] = coxae.mean(0) + 1.5 * (coxae - coxae.mean(0))
    _, s_free, _ = mesh._body_transform(pts)
    _, s_fixed, _ = mesh._body_transform(pts, fixed_scale=0.9)
    assert s_free == pytest.approx(1.5, rel=1e-3)  # per-frame fit follows the spread
    assert s_fixed == 0.9  # fixed scale ignores it
    # the rigid body's extent then tracks the held scale, not the per-frame one: both
    # poses share the fit rotation, so the body spans scale exactly as 0.9 / 1.5.
    body = mesh.vert_slot == 0
    span_fixed = np.ptp(mesh.pose(pts, body_scale=0.9)[0][body], axis=0)
    span_free = np.ptp(mesh.pose(pts)[0][body], axis=0)
    np.testing.assert_allclose(span_fixed / span_free, 0.9 / 1.5, rtol=1e-6)


def test_nmf_mesh_hidden_face_mask_hides_parts():
    """``hidden_face_mask`` selects exactly the faces of the named body parts."""
    from deeperfly.inverse_kinematics.mesh import load_nmf_mesh

    mesh = load_nmf_mesh()
    assert "wings" in mesh.part_names  # the baked asset carries part labels
    none_hidden = mesh.hidden_face_mask([])
    assert not none_hidden.any() and none_hidden.shape == (mesh.faces.shape[0],)
    wings = mesh.hidden_face_mask(["wings"])
    assert wings.any()
    # the wing faces are exactly the wing-part faces (each face is one part)
    face_part = mesh.vert_part[mesh.faces[:, 0]]
    expected = face_part == mesh.part_names.index("wings")
    np.testing.assert_array_equal(wings, expected)
    # an unknown part name hides nothing
    assert not mesh.hidden_face_mask(["nonexistent"]).any()


# -- head / abdomen articulation chains ---------------------------------------


def _place_chain_markers(chain, theta, sim, pts, index, size=1.0):
    """Fill ``pts`` with a chain's markers, FK'd by ``theta`` then placed by ``sim``.

    ``size`` grows the neutral markers about the chain base before the FK, the same
    way ``solve_chain(scale=...)`` and the overlay mesh do, so a recording can be
    synthesized for a head/abdomen larger or smaller than the model geometry.
    """
    from deeperfly.inverse_kinematics.articulation import chain_affine

    rot, scale, trans = sim
    base = np.asarray(chain.anchors[0], dtype=float)
    for k, (name, depth) in enumerate(zip(chain.marker_names, chain.marker_depth)):
        a, b = chain_affine(chain, depth, theta)
        neutral = base + size * (chain.marker_neutral[k] - base)
        model = a @ neutral + b
        pts[:, index[name]] = scale * (rot @ model) + trans


def test_chain_fk_matches_numpy_affine():
    """The JAX chain FK agrees with the numpy ``chain_affine`` used to pose the mesh."""
    from deeperfly.inverse_kinematics.articulation import (
        chain_affine,
        load_articulation,
    )
    from deeperfly.inverse_kinematics.kinematics import make_chain_fk

    rng = np.random.default_rng(1)
    for chain in load_articulation().chains:
        fk = make_chain_fk(chain.anchors, chain.axes, chain.marker_depth)
        theta = _bent_angles(chain, rng)
        jax_markers = np.asarray(fk(theta, chain.marker_neutral))
        for k, depth in enumerate(chain.marker_depth):
            a, b = chain_affine(chain, depth, theta)
            np.testing.assert_allclose(
                jax_markers[k], a @ chain.marker_neutral[k] + b, atol=1e-6
            )


@pytest.mark.parametrize(
    "name,size",
    [("head", 1.0), ("abdomen", 1.0), ("head", 1.6), ("abdomen", 0.7)],
)
def test_solve_chain_recovers_known_angles(name, size):
    """A bounded-LS chain solve recovers the angles that generated the markers.

    Also exercises a head/abdomen ``size`` other than the model's: the markers are
    placed for a chain grown/shrunk by ``size`` and the solve is told that size, so
    it must still recover the generating angles and reach the markers.
    """
    from deeperfly.inverse_kinematics.articulation import load_articulation
    from deeperfly.inverse_kinematics.core import solve_chain

    fly = Skeleton.fly()
    index = {n: i for i, n in enumerate(fly.point_names)}
    chain = load_articulation().chain(name)
    rng = np.random.default_rng(0)
    sim = (_rot_z(0.4), 1.6, np.array([2.0, -1.0, 3.0]))
    truth = _bent_angles(chain, rng)
    pts = np.full((3, fly.n_points, 3), np.nan)
    _place_chain_markers(chain, truth, sim, pts, index, size=size)

    angles, world = solve_chain(
        pts, index, chain, sim, max_nfev=300, regularization=0.0, scale=size
    )
    np.testing.assert_allclose(angles[0], truth, atol=1e-3)
    # the fitted markers reproject onto the (noise-free) measurements
    for k, mname in enumerate(chain.marker_names):
        np.testing.assert_allclose(world[0, k], pts[0, index[mname]], atol=1e-5)


def test_abdomen_bounds_are_downward_only():
    """The baked abdomen chain only bends ventrally (downward), <=30 deg per joint.

    The few midline markers under-constrain the 5-DOF sagittal chain, so symmetric
    limits let the solver fold it into a non-physical zig-zag; a downward-only,
    monotone range keeps the fit a smooth ventral curl (model +pitch is dorsal, so
    "down" is the negative range).
    """
    from deeperfly.inverse_kinematics.articulation import load_articulation

    lo, hi = load_articulation().chain("abdomen").bounds
    assert np.all(hi <= 1e-9)  # no dorsal (upward) flexion
    assert np.all(lo >= np.deg2rad(-30) - 1e-9)  # at most 30 deg per joint
    assert np.all(lo < hi)  # a real (non-empty) downward range


def test_articulation_bakes_attachment_body_frames():
    """The asset carries a body frame for every chain body a marker can attach to."""
    from deeperfly.inverse_kinematics.articulation import load_articulation

    art = load_articulation()
    assert art.bodies, "the regenerated asset must carry attachment-body frames"
    # the default markers' attachment bodies are all present, on the right chain
    assert art.bodies["c_abdomen3"]["chain"] == "abdomen"
    assert art.bodies["l_pedicel"]["chain"] == "head"
    for body in ("c_abdomen3", "c_abdomen5", "c_abdomen6", "l_pedicel"):
        assert body in art.bodies


def test_articulation_marker_override_recomputes_neutral_from_offset():
    """A config marker override places its neutral at ``body_frame @ offset``."""
    from deeperfly.inverse_kinematics.articulation import Articulation

    over = {
        "abdomen": {
            "l_abdomen0": {"body": "c_abdomen3", "offset": [0.0, 0.05, 0.5]},
            "r_abdomen0": {"body": "c_abdomen3", "offset": [0.0, -0.05, 0.5]},
        }
    }
    art = Articulation.load(marker_overrides=over)
    ab = art.chain("abdomen")
    assert ab.marker_names == ("l_abdomen0", "r_abdomen0")  # table replaces the set
    assert ab.marker_depth == (2, 2)  # depth follows from the attachment body
    frame = art.bodies["c_abdomen3"]
    pos = np.asarray(frame["pos"])
    mat = np.asarray(frame["mat"]).reshape(3, 3)
    np.testing.assert_allclose(
        ab.marker_neutral[0], pos + mat @ np.array([0.0, 0.05, 0.5]), atol=1e-7
    )


def test_articulation_marker_override_rejects_bad_body():
    """An unknown / wrong-chain attachment body is a clear config error."""
    from deeperfly.inverse_kinematics.articulation import Articulation

    with pytest.raises(ValueError, match="unknown body"):
        Articulation.load(
            marker_overrides={"abdomen": {"x": {"body": "nope", "offset": [0, 0, 0]}}}
        )
    with pytest.raises(ValueError, match="belongs to"):
        # l_pedicel is a head body, not an abdomen one
        Articulation.load(
            marker_overrides={
                "abdomen": {"x": {"body": "l_pedicel", "offset": [0, 0, 0]}}
            }
        )


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("name,size", [("head", 1.7), ("abdomen", 0.7)])
def test_estimate_chain_scale_recovers_contour_length(name, size):
    """The size estimate recovers a chain grown/shrunk by ``size`` from its contour.

    The estimate is the measured/model ratio of the chain's contour length
    (``base -> per-depth marker centroids``). The head's three DOFs share one pivot,
    so that length is exactly rotation-invariant and is checked at a bent pose; the
    abdomen is a serial chain whose contour is exact at the neutral (straight) pose,
    where it scales precisely with ``size``.
    """
    from deeperfly.inverse_kinematics.articulation import (
        estimate_chain_scale,
        load_articulation,
    )

    fly = Skeleton.fly()
    index = {n: i for i, n in enumerate(fly.point_names)}
    chain = load_articulation().chain(name)
    rng = np.random.default_rng(5)
    rot, body_scale, trans = _rot_z(0.3), 1.4, np.array([1.0, -2.0, 0.5])
    theta = (
        _bent_angles(chain, rng) if name == "head" else np.zeros(len(chain.dof_names))
    )
    pts = np.full((3, fly.n_points, 3), np.nan)
    _place_chain_markers(chain, theta, (rot, body_scale, trans), pts, index, size=size)

    cols = [index[m] for m in chain.marker_names]
    world = np.stack([pts[:, c] for c in cols], axis=1)
    local = ((world - trans) @ rot) / body_scale
    assert estimate_chain_scale(local, chain) == pytest.approx(size, abs=1e-6)


def test_estimate_chain_scale_defaults_to_one_when_unobserved():
    """With no frame where the contour is observed, the size stays at the model (1.0)."""
    from deeperfly.inverse_kinematics.articulation import (
        estimate_chain_scale,
        load_articulation,
    )

    chain = load_articulation().chain("abdomen")
    local = np.full((4, len(chain.marker_names), 3), np.nan)
    assert estimate_chain_scale(local, chain) == 1.0


def test_solve_inverse_kinematics_includes_head_and_abdomen(fly):
    """With an articulation, the solve adds head/abdomen DOFs and fills their markers."""
    from deeperfly.inverse_kinematics.articulation import load_articulation

    template = KinematicTemplate.load("neuromechfly")
    art = load_articulation()
    index = {n: i for i, n in enumerate(fly.point_names)}
    sim = (_rot_z(0.2), 1.5, np.array([1.0, 2.0, -1.0]))
    rng = np.random.default_rng(3)

    pts = np.full((2, fly.n_points, 3), np.nan)
    # place the coxae (registers the body) and the head/abdomen markers
    from deeperfly.inverse_kinematics.mesh import load_nmf_mesh

    kpn = load_nmf_mesh().kp_neutral
    for cname in art.coxa_points:
        pts[:, index[cname]] = sim[1] * (sim[0] @ kpn[index[cname]]) + sim[2]
    truth = {c.name: _bent_angles(c, rng, frac=(0.4, 0.6)) for c in art.chains}
    for c in art.chains:
        _place_chain_markers(c, truth[c.name], sim, pts, index)

    res = solve_inverse_kinematics(
        pts, fly, template, articulation=art, regularization=0.0
    )
    assert "c_thorax-c_head-yaw" in res.angle_names
    assert "c_abdomen12-c_abdomen3-pitch" in res.angle_names
    # the antenna + abdomen markers are filled in the reprojected model points
    for name in ("l_antenna", "r_antenna", "l_abdomen0", "r_abdomen2"):
        assert np.isfinite(res.model_pts3d[0, index[name]]).all()


def test_nmf_mesh_articulates_head_and_abdomen_from_angles():
    """Passing chain angles bends the head/abdomen mesh nodes; omitting them is rigid."""
    from deeperfly.inverse_kinematics.articulation import load_articulation
    from deeperfly.inverse_kinematics.mesh import load_nmf_mesh

    mesh = load_nmf_mesh()
    art = load_articulation()
    names = art.dof_names
    angles = np.zeros(len(names))
    angles[names.index("c_thorax-c_head-yaw")] = np.deg2rad(35)
    angles[names.index("c_abdomen5-c_abdomen6-pitch")] = np.deg2rad(-40)

    rigid, _ = mesh.pose(mesh.kp_neutral)
    bent, _ = mesh.pose(mesh.kp_neutral, angles, names)
    # head + abdomen node vertices move; passing no angles reproduces the rigid pose.
    node = np.isin(mesh.vert_slot, np.flatnonzero(mesh.slot_chain >= 0))
    assert np.nanmax(np.abs(bent[node] - rigid[node])) > 0.05
    body = mesh.vert_slot == 0
    np.testing.assert_allclose(bent[body], rigid[body], atol=1e-9)
