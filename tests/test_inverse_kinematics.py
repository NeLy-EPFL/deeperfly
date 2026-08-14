"""Tests for the inverse-kinematics model: template, alignment, articulation, mesh.

Everything here is solver-free -- the model *geometry* and the measurements deeperfly
takes from a recording, none of which needs the optional QuickIK extra. The solve
itself lives in ``test_inverse_kinematics_quickik.py``, and the body plan and forward
kinematics in ``test_ik_forward_bodyplan.py``.

The split is not cosmetic: the mesh overlay, the reprojected NMF skeleton and the GUI's
static-fit path all run on a plain install, so the code they need must be tested
without the extra installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import (
    IK_SEGLENS as _SEGLENS,
)
from helpers import (
    bent_angles as _bent_angles,
)
from helpers import (
    place_chain_markers as _place_chain_markers,
)
from helpers import (
    rot_z as _rot_z,
)
from helpers import synth_leg_pose as _synth_pose

from deeperfly.inverse_kinematics.align import body_alignment, to_local, to_world
from deeperfly.inverse_kinematics.template import KinematicTemplate
from deeperfly.skeleton import Skeleton


@pytest.fixture
def template() -> KinematicTemplate:
    return KinematicTemplate.load("neuromechfly")


@pytest.fixture
def fly() -> Skeleton:
    return Skeleton.fly()


# -- alignment ---------------------------------------------------------------


def test_alignment_frame_is_orthonormal(template, fly, rng):
    """The body frame recovered from the coxae is a proper rotation, however posed."""
    pts3d, _ = _synth_pose(template, fly, rng, r_body=_rot_z(0.5))
    rb = body_alignment(pts3d, fly, template).r_body
    np.testing.assert_allclose(rb.T @ rb, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(rb), 1.0, atol=1e-6)


def test_to_local_to_world_round_trip(rng):
    r = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    origin = rng.normal(size=3)
    pts = rng.normal(size=(5, 3))
    back = to_world(to_local(pts, origin, r), origin, r)
    np.testing.assert_allclose(back, pts, atol=1e-10)


def test_measured_seglens_recovered(template, fly, rng):
    """Bone lengths are measured back off the data -- what the body plan is built from."""
    pts3d, _ = _synth_pose(template, fly, rng)
    align = body_alignment(pts3d, fly, template)
    np.testing.assert_allclose(align.seglens["rf"], _SEGLENS, atol=1e-6)


# -- the IK stage's own constant-point pin (superseded by [postprocess]) ------


def test_freeze_3d_collapses_to_temporal_median():
    """A jittered column is replaced by its temporal nanmedian; NaN frames filled."""
    from deeperfly.postprocess import freeze_3d

    rng = np.random.default_rng(0)
    pts3d = rng.normal(size=(20, 4, 3))
    truth = np.array([5.0, -2.0, 1.0])
    pts3d[:, 1] = truth + rng.normal(scale=0.1, size=(20, 3))
    pts3d[3, 1] = np.nan  # occluded frames
    pts3d[7, 1] = np.nan
    expected = np.nanmedian(pts3d[:, 1], axis=0)

    out = freeze_3d(pts3d, [1])
    assert np.isfinite(out[:, 1]).all()  # occluded frames get filled by the median
    np.testing.assert_allclose(out[:, 1], expected[None].repeat(20, 0), atol=1e-12)
    np.testing.assert_array_equal(out[:, 0], pts3d[:, 0])  # other points untouched
    assert np.isnan(pts3d[3, 1]).all()  # input array is not mutated


def test_freeze_3d_all_nan_column_stays_nan():
    """A point that is never observed stays all-NaN (its leg is skipped downstream)."""
    from deeperfly.postprocess import freeze_3d

    pts3d = np.zeros((5, 3, 3))
    pts3d[:, 2] = np.nan
    out = freeze_3d(pts3d, [2])
    assert np.isnan(out[:, 2]).all()


def test_the_ik_pin_says_when_it_is_redundant_and_when_it_is_invisible(fly, caplog):
    """The two things a reader of a config cannot see for themselves.

    ``[inverse_kinematics].constant_points`` is superseded by ``{ op = "static" }``, and
    the interesting cases are the two ways the two lists can disagree. A point in both is
    pinned twice, which is a harmless no-op (the median of a constant is that constant)
    but worth saying. A point in the IK list ALONE means the fit runs on a pose no stage
    output records -- so the stored angles and the stored 3D disagree for it, silently,
    which is the case this warning exists for.
    """
    from deeperfly.config import Config
    from deeperfly.pipeline.stages import _pin_for_fit

    pts3d = np.zeros((6, len(fly.point_names), 3))
    config = Config.from_dict(
        {"postprocess": {"ops": [{"op": "static", "points": ["lf_thorax_coxa"]}]}}
    )
    with caplog.at_level("INFO", logger="deeperfly"):
        _pin_for_fit(config, fly, pts3d, ["lf_thorax_coxa", "rf_thorax_coxa"])
    text = caplog.text
    assert "already frozen" in text and "lf_thorax_coxa" in text
    assert "NO stage output records" in text and "rf_thorax_coxa" in text


def test_the_ik_pin_names_its_own_key_on_a_typo(fly):
    """Two config keys carry a held-still list; the error has to say which one."""
    from deeperfly.config import Config
    from deeperfly.pipeline.stages import _pin_for_fit

    pts3d = np.zeros((4, len(fly.point_names), 3))
    with pytest.raises(ValueError, match=r"\[inverse_kinematics\]\.constant_points"):
        _pin_for_fit(Config.from_dict({}), fly, pts3d, ["not_a_point"])


# -- confidence weights ------------------------------------------------------


def test_confidence_weights_average_over_views():
    """A 3D point's solve weight is the mean confidence of the 2D views behind it."""
    from deeperfly.pipeline.stages import _confidence_weights

    assert _confidence_weights(None) is None
    conf = np.array([[[0.2, 1.0]], [[0.8, np.nan]]])  # (V=2, T=1, P=2)
    got = _confidence_weights(conf)
    np.testing.assert_allclose(got, [[0.5, 1.0]])  # NaN views are ignored, not zeros


def test_confidence_weights_all_nan_point_is_nan():
    """A point unseen in every view yields NaN, which the solve reads as unobserved."""
    from deeperfly.pipeline.stages import _confidence_weights

    got = _confidence_weights(np.full((2, 1, 1), np.nan))
    assert np.isnan(got).all()


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
