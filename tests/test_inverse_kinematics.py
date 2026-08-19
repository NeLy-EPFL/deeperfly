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
from helpers import deepfly3d_skeleton, fly38_skeleton  # noqa: F401
from helpers import (
    place_chain_markers as _place_chain_markers,
)
from helpers import (
    rot_z as _rot_z,
)
from helpers import synth_leg_pose as _synth_pose

from deeperfly.inverse_kinematics.align import (
    body_alignment,
    mirror_leg_pairs,
    symmetrize_seglens,
    to_local,
    to_world,
)
from deeperfly.inverse_kinematics.template import KinematicTemplate
from deeperfly.skeleton import Skeleton


@pytest.fixture
def template() -> KinematicTemplate:
    return KinematicTemplate.load("neuromechfly")


@pytest.fixture
def fly() -> Skeleton:
    return deepfly3d_skeleton()


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


# -- left/right segment symmetry ---------------------------------------------


def _asymmetric_pose(template, fly, rng, *, stretch=1.10):
    """A synthetic pose whose LEFT legs are ``stretch`` times longer than its right.

    The bones are the ones a real recording gets wrong: both sides are the same animal,
    but each is triangulated from its own camera triplet, so one side comes back
    systematically longer. Stretching about each leg's own coxa leaves the coxae -- and
    therefore the body frame and the registration -- exactly where the symmetric pose
    put them, so the only thing under test is the lengths.
    """
    pts3d, truth = _synth_pose(template, fly, rng)
    index = {n: i for i, n in enumerate(fly.point_names)}
    for leg in template.legs:
        if leg.side != "l":
            continue
        coxa = pts3d[:, index[leg.point_names[0]]]
        for name in leg.point_names[1:]:
            col = index[name]
            pts3d[:, col] = coxa + stretch * (pts3d[:, col] - coxa)
    return pts3d, truth


def test_mirror_leg_pairs_come_from_the_skeletons_declared_symmetries(template, fly):
    """The pairing is derived from ``[skeleton].symmetries``, not from the leg names."""
    assert set(map(frozenset, mirror_leg_pairs(template, fly))) == {
        frozenset(("lf", "rf")),
        frozenset(("lm", "rm")),
        frozenset(("lh", "rh")),
    }


def test_a_leg_whose_partner_is_not_fitted_has_no_pair(fly):
    """``legs = [...]`` may name one side of a pair; that leg simply keeps its own bones."""
    half = KinematicTemplate.load("neuromechfly", legs=["rf", "lf", "rm"])
    assert set(map(frozenset, mirror_leg_pairs(half, fly))) == {frozenset(("lf", "rf"))}


def test_a_skeleton_declaring_no_symmetries_symmetrizes_nothing(
    template, fly, rng, caplog
):
    """Declaring no pairs means "this subject is not bilaterally symmetric" -- obey it.

    The same convention :meth:`Skeleton.flip_perm` and the chirality QC follow. Silence
    would be wrong here: the config asked for symmetric segments and did not get them.
    """
    from dataclasses import replace

    asymmetric = replace(fly, symmetries=np.empty((0, 2), np.int64))
    pts3d, _ = _asymmetric_pose(template, asymmetric, rng)
    assert mirror_leg_pairs(template, asymmetric) == ()
    with caplog.at_level("WARNING", logger="deeperfly"):
        align = body_alignment(pts3d, asymmetric, template, symmetric_segments=True)
    assert "no leg" in caplog.text and "symmetric_segments" in caplog.text
    assert align.seglens["lf"][2] > align.seglens["rf"][2]


def test_symmetric_segments_gives_a_mirror_pair_one_shared_bone(template, fly, rng):
    """Both sides end on the mean of the two measurements -- neither side's own value."""
    pts3d, _ = _asymmetric_pose(template, fly, rng, stretch=1.10)
    measured = body_alignment(pts3d, fly, template).seglens
    shared = body_alignment(pts3d, fly, template, symmetric_segments=True).seglens

    for left, right in (("lf", "rf"), ("lm", "rm"), ("lh", "rh")):
        np.testing.assert_allclose(shared[left], shared[right], atol=1e-12)
        np.testing.assert_allclose(
            shared[left][1:],
            0.5 * (measured[left][1:] + measured[right][1:]),
            atol=1e-12,
        )
        # The mean of a 10%-stretched side and an unstretched one, so strictly between.
        assert (shared[left][1:] < measured[left][1:]).all()
        assert (shared[left][1:] > measured[right][1:]).all()


def test_symmetry_does_not_privilege_a_side_and_leaves_the_input_alone(
    template, fly, rng
):
    """Averaging into one side would import that side's error into both.

    The same argument :func:`~deeperfly.postprocess.symmetrize_3d` makes for the pose.
    Swapping which side is stretched must give the identical answer.
    """
    left_long, _ = _asymmetric_pose(template, fly, rng, stretch=1.10)
    measured = body_alignment(left_long, fly, template).seglens
    before = {k: v.copy() for k, v in measured.items()}
    a = symmetrize_seglens(measured, template, fly)
    swapped = {
        "lf": measured["rf"],
        "rf": measured["lf"],
        "lm": measured["rm"],
        "rm": measured["lm"],
        "lh": measured["rh"],
        "rh": measured["lh"],
    }
    b = symmetrize_seglens(swapped, template, fly)
    for leg in ("lf", "rf", "lm", "rm", "lh", "rh"):
        np.testing.assert_allclose(a[leg], b[leg], atol=1e-12)
    for leg, arr in before.items():  # the caller's mapping is not modified
        np.testing.assert_array_equal(measured[leg], arr)


def test_a_segment_seen_on_one_side_only_borrows_its_mirror(template, fly, rng):
    """Better than the alternative: an unmeasured bone otherwise gets the MODEL's own.

    Without symmetry a never-triangulated segment measures 0 and
    ``_model_seglens`` substitutes the generic model bone. With it, the same segment on
    the other side of the same animal is available and is the better answer -- so the
    missing side adopts it rather than averaging a zero in.
    """
    pts3d, _ = _synth_pose(template, fly, rng)
    index = {n: i for i, n in enumerate(fly.point_names)}
    rf = next(leg for leg in template.legs if leg.name == "rf")
    pts3d[:, index[rf.point_names[3]]] = np.nan  # rf tibia-tarsus never triangulated

    measured = body_alignment(pts3d, fly, template).seglens
    assert measured["rf"][3] == 0.0 and measured["rf"][4] == 0.0  # both its bones gone
    shared = body_alignment(pts3d, fly, template, symmetric_segments=True).seglens
    np.testing.assert_allclose(shared["rf"][3], measured["lf"][3], atol=1e-12)
    np.testing.assert_allclose(shared["rf"][4], measured["lf"][4], atol=1e-12)
    # ...and the observed side keeps its own measurement rather than being halved.
    np.testing.assert_allclose(shared["lf"], measured["lf"], atol=1e-12)


def test_a_segment_seen_on_neither_side_is_left_for_the_model_fallback(
    template, fly, rng
):
    """0.0 is the "never observed" sentinel ``_model_seglens`` reads; keep it."""
    pts3d, _ = _synth_pose(template, fly, rng)
    index = {n: i for i, n in enumerate(fly.point_names)}
    for leg in ("lf", "rf"):
        chain = next(x for x in template.legs if x.name == leg)
        pts3d[:, index[chain.point_names[4]]] = np.nan  # both claws gone

    shared = body_alignment(pts3d, fly, template, symmetric_segments=True).seglens
    assert shared["lf"][4] == 0.0 and shared["rf"][4] == 0.0


def test_symmetric_segments_reports_the_gap_it_closed(template, fly, rng, caplog):
    """The premise-check: sides already agreeing had nothing to share."""
    pts3d, _ = _asymmetric_pose(template, fly, rng, stretch=1.10)
    with caplog.at_level("INFO", logger="deeperfly"):
        body_alignment(pts3d, fly, template, symmetric_segments=True)
    assert "3 mirror leg pair(s)" in caplog.text
    assert "9.5%" in caplog.text  # |1.10 - 1| / mean(1.10, 1) worst per pair


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


def test_abdomen_bends_vertically_and_laterally_but_never_twists():
    """The abdomen chain bends in two planes per joint and has no axial-twist DOF.

    Vertical (``pitch``, about ``+Y``) is downward-only: the few midline markers
    under-constrain the sagittal chain, so symmetric limits let the solver fold it into a
    non-physical zig-zag, and a downward-only range keeps it a smooth ventral curl (model
    +pitch is dorsal, so "down" is the negative range). Lateral is symmetric, because a
    fly bends either way.

    The lateral DOF is flygym's ``roll``, and the twist that is excluded is flygym's
    ``yaw`` -- the names are anatomically rotated on this chain, which is exactly why this
    test exists. Measured off the MJCF, every hinge's axes are, in the child segment's own
    frame: ``pitch`` = local *y* (sagittal), ``roll`` = local *z* (lateral), ``yaw`` =
    local *x* (the axial twist). See ``docs/explanation/keypoints.md``.

    Twist is excluded by construction, and the thing to assert is that no axis lies along
    the chain's own long axis -- **not** that no axis has a world ``X`` component. The
    lateral axes legitimately carry one: each segment's local *z* is tilted out of world
    dorsal by its place in the resting curl (0.22-0.34 of world x), and dropping that tilt
    for a round ``[0, 0, 1]`` would be a different, worse model. Successive bends can also
    compose to a net axial rotation -- that is geometry, not a degree of freedom.
    """
    from deeperfly.inverse_kinematics.articulation import load_articulation

    chain = load_articulation().chain("abdomen")
    lo, hi = chain.bounds
    axes = np.asarray(chain.axes, dtype=float)
    anchors = np.asarray(chain.anchors, dtype=float)

    # The long axis is the direction the chain runs; no joint may rotate about it.
    long_axis = anchors[-1] - anchors[0]
    long_axis /= np.linalg.norm(long_axis)
    # The real axes clear it with room to spare: the worst is 0.159 (the waist's lateral
    # hinge, the most tilted segment). A twist DOF would read ~1.0 here.
    assert np.abs(axes @ long_axis).max() < 0.2, "a joint rotates about the long axis"

    pitch = [i for i, n in enumerate(chain.dof_names) if n.endswith("pitch")]
    lateral = [i for i, n in enumerate(chain.dof_names) if n.endswith("roll")]
    assert len(pitch) == len(lateral) > 0, "every joint bends in both planes"
    # flygym's `yaw` IS the twist on this chain, so its absence is the assertion.
    assert not [n for n in chain.dof_names if n.endswith("yaw")], (
        "a twist DOF is present"
    )

    # vertical: downward only, at most 30 deg per joint
    assert np.all(hi[pitch] <= 1e-9)
    assert np.all(lo[pitch] >= np.deg2rad(-30) - 1e-9)
    # lateral: symmetric about zero
    np.testing.assert_allclose(lo[lateral], -hi[lateral], atol=1e-12)
    assert np.all(hi[lateral] > 0)
    assert np.all(lo < hi)  # every range is real (non-empty)


def test_abdomen_pitch_is_vertical_and_roll_is_lateral():
    """The two DOF families move the markers in the planes the model says they should.

    Named axes are easy to get backwards, and the consequence -- an abdomen that swings
    sideways when the fly nodded it down -- is invisible in a residual. So drive each
    family on its own and check which coordinate moves. Doubly worth pinning here because
    flygym's names are anatomically rotated on this chain: it is ``roll`` that swings the
    abdomen laterally, not ``yaw`` (which is the twist, and is excluded).
    """
    from deeperfly.inverse_kinematics.articulation import (
        chain_markers,
        load_articulation,
    )

    chain = load_articulation().chain("abdomen")
    zero = np.zeros(len(chain.dof_names))
    rest = chain_markers(chain, zero, 1.0)

    def moved(which: str, deg: float) -> np.ndarray:
        theta = zero.copy()
        for i, n in enumerate(chain.dof_names):
            if n.endswith(which):
                theta[i] = np.deg2rad(deg)
        return np.abs(chain_markers(chain, theta, 1.0) - rest).max(axis=0)

    vertical = moved("pitch", -15.0)
    lateral = moved("roll", 10.0)
    assert vertical[2] > 10 * max(vertical[1], 1e-12), (
        "pitch must move the markers in z"
    )
    # Lateral motion is y-dominant but not purely y: each hinge axis is the segment's
    # own local z, tilted by its place in the resting curl, so a little z comes with
    # it. Measured 0.544 in y against 0.051 in z -- assert the ratio, not purity.
    assert lateral[1] == max(lateral), "roll must move the markers mostly in y"
    assert lateral[2] < 0.15 * lateral[1], "roll leaked into z"


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
    # Depth follows from the attachment body, and is a count of proximal JOINTS -- so it
    # tracks the chain's DOF count rather than its segment count. Read from the body
    # rather than written down, or adding a DOF per segment desyncs the two silently.
    body_depth = int(art.bodies["c_abdomen3"]["depth"])
    assert ab.marker_depth == (body_depth, body_depth)
    assert body_depth <= len(ab.dof_names)
    frame = art.bodies["c_abdomen3"]
    pos = np.asarray(frame["pos"])
    mat = np.asarray(frame["mat"]).reshape(3, 3)
    np.testing.assert_allclose(
        ab.marker_neutral[0], pos + mat @ np.array([0.0, 0.05, 0.5]), atol=1e-7
    )


def test_a_marker_table_can_carry_the_base_nomination_over():
    """Redeclaring a chain must not silently drop its base marker.

    The table *replaces* the chain's markers, so a config that retargets the head would
    otherwise put it back on the registered base -- a change of pivot with nothing in the
    config that says so. ``base = true`` is how the nomination is expressed, and the base
    is forced to depth 0 whatever its attachment body's depth says (``c_head``'s is 3),
    because a point on the rotation center is moved by none of the chain's DOFs.
    """
    from deeperfly.inverse_kinematics.articulation import Articulation

    table = {
        "head": {
            "neck": {"body": "c_head", "offset": [0.0, 0.0, 0.0], "base": True},
            "l_antenna": {"body": "l_pedicel", "offset": [0.0, 0.0, 0.0]},
            "r_antenna": {"body": "r_pedicel", "offset": [0.0, 0.0, 0.0]},
        }
    }
    head = Articulation.load(marker_overrides=table).chain("head")
    assert head.base_point == "neck"
    assert head.marker_depth[head.marker_index("neck")] == 0
    np.testing.assert_allclose(
        head.marker_neutral[head.marker_index("neck")], head.anchors[0], atol=1e-7
    )

    forgotten = {k: v for k, v in table["head"].items() if k != "neck"}
    assert (
        Articulation.load(marker_overrides={"head": forgotten}).chain("head").base_point
        is None
    )


def test_a_marker_table_rejects_two_bases_and_a_base_that_is_not_a_marker():
    from deeperfly.inverse_kinematics.articulation import Articulation

    with pytest.raises(ValueError, match="a chain has one base"):
        Articulation.load(
            marker_overrides={
                "head": {
                    "l_antenna": {
                        "body": "l_pedicel",
                        "offset": [0, 0, 0],
                        "base": True,
                    },
                    "r_antenna": {
                        "body": "r_pedicel",
                        "offset": [0, 0, 0],
                        "base": True,
                    },
                }
            }
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


@pytest.mark.parametrize(
    "name,size,has_ruler", [("head", 1.7, True), ("abdomen", 0.7, False)]
)
def test_a_resized_chain_is_recovered_at_a_bent_pose(name, size, has_ruler):
    """A chain grown/shrunk by ``size`` is measured back -- while it is BENT.

    The bent pose is the point of the test. A *ruler* only ever compares distances the
    chain's own joints cannot change, so articulating the chain must not move it at all.
    That is a real property and not a formality: the abdomen's markers sit on the dorsal
    *surface*, the outside of a ventral bend, so a polyline through them lengthens by 57%
    over the joints' range -- a ruler drawn along the chain would read this bent pose as a
    much bigger animal.

    The two chains take the two different routes, which is why ``has_ruler`` is a
    parameter rather than an assumption. The **head** has an invariant ruler (the
    neck-to-antennae radius) and :func:`estimate_chain_scale` recovers the size outright.
    The **abdomen** has none -- deliberately, since its one qualifying pair only qualified
    because two markers shared a body -- so the ruler falls back to model size and
    :func:`calibrate_chain` recovers it instead, by fitting posture and size together.

    Each chain is measured against the skeleton that labels its markers, and is
    synthesized at a base the model does not share, so this also pins that the measurement
    is internal to the chain: measured from the model's own anchor, that shift would read
    as a change in size.
    """
    from deeperfly.inverse_kinematics.articulation import (
        calibrate_chain,
        estimate_chain_scale,
        load_articulation,
    )

    fly = fly38_skeleton()
    index = {n: i for i, n in enumerate(fly.point_names)}
    chain = load_articulation().chain(name)
    rng = np.random.default_rng(5)
    rot, body_scale, trans = _rot_z(0.3), 1.4, np.array([1.0, -2.0, 0.5])
    pts = np.full((3, fly.n_points, 3), np.nan)
    _place_chain_markers(
        chain,
        _bent_angles(chain, rng),
        (rot, body_scale, trans),
        pts,
        index,
        size=size,
        shift=np.array([0.05, -0.02, -0.15]),
    )

    cols = [index.get(m, -1) for m in chain.marker_names]
    world = np.stack(
        [pts[:, c] if c >= 0 else np.full((3, 3), np.nan) for c in cols], axis=1
    )
    local = ((world - trans) @ rot) / body_scale

    ruler = estimate_chain_scale(local, chain, warn_if_unmeasurable=False)
    if has_ruler:
        assert ruler == pytest.approx(size, abs=1e-6)
    else:
        assert ruler == 1.0, (
            "a chain with no invariant pair must fall back to model size"
        )
    fitted, _ = calibrate_chain(local, chain, seed_scale=ruler, fit_root=True)
    assert fitted == pytest.approx(size, rel=2e-3)


def test_the_size_ruler_is_only_ever_an_invariant_separation():
    """Each chain's ruler is the marker separation its own joints cannot change.

    Pins *which* distances are measured, because that is the whole design: the head's
    is the neck-to-antennae radius (rotating about a point cannot change a distance from
    that point) and the abdomen's is its one pair sharing a body. Every other abdomen
    pair spans a hinge and is rejected -- including the neighbouring ones a chain-length
    ruler would naturally string together.
    """
    from deeperfly.inverse_kinematics.articulation import (
        _midline_groups,
        _rigid_ruler,
        load_articulation,
    )

    art = load_articulation()
    picked = {}
    for chain in art.chains:
        groups = _midline_groups(chain)
        names = [tuple(chain.marker_names[c] for c in g) for g in groups]
        picked[chain.name] = {
            frozenset(names[i] + names[j]) for i, j, _ in _rigid_ruler(chain, groups)
        }
    assert picked["head"] == {frozenset({"neck", "l_antenna", "r_antenna"})}
    # The abdomen has NO ruler, and that is the design rather than a gap. Its only
    # qualifying pair used to be abdomen3-abdomen4, which qualified solely because both
    # markers hung off c_abdomen6 -- so the model held them rigidly apart while a real
    # abdomen does not, 18% short of a measured fly and unreachable by any angle. Moving
    # each stripe one segment proximal put a hinge between them, and removed the residual
    # and the ruler together. `calibrate_chain` measures the size without one.
    assert picked["abdomen"] == set()


def test_the_size_ruler_ignores_a_left_right_split():
    """Pushing a mirror pair symmetrically apart must not change the measured size.

    A left-to-right distance carries the two sides' triangulation disagreement at full
    strength, and on real data that is not a rounding error: across the corpus the
    ``l_antenna``-``r_antenna`` span reads 1.515x the model where either antenna's
    distance to the ``neck`` reads 1.196x -- 26% wider, the same sign in all 30
    recordings, because each antenna is triangulated from its own side's cameras. Folding
    a mirror pair to its midpoint before measuring is what makes the ruler immune, and
    this pins that it is.
    """
    from deeperfly.inverse_kinematics.articulation import (
        estimate_chain_scale,
        load_articulation,
    )

    fly = fly38_skeleton()
    index = {n: i for i, n in enumerate(fly.point_names)}
    chain = load_articulation().chain("head")
    rng = np.random.default_rng(11)
    pts = np.full((3, fly.n_points, 3), np.nan)
    _place_chain_markers(
        chain, _bent_angles(chain, rng), (np.eye(3), 1.0, np.zeros(3)), pts, index
    )
    cols = [index[m] for m in chain.marker_names]
    local = pts[:, cols]

    left = chain.marker_names.index("l_antenna")
    right = chain.marker_names.index("r_antenna")
    spread = local.copy()
    # Push the pair 30% apart along its own axis -- symmetric about the midpoint, which
    # is how a two-sided reconstruction splits, and frame-independent (the head here is
    # bent, so "along y" would not be symmetric in the head's own frame).
    mid = 0.5 * (local[:, left] + local[:, right])
    for k in (left, right):
        spread[:, k] = mid + 1.3 * (local[:, k] - mid)
    moved = np.linalg.norm(spread[:, left] - local[:, left], axis=-1)
    assert moved.min() > 0.02, "the pair really moved"
    assert estimate_chain_scale(spread, chain) == pytest.approx(
        estimate_chain_scale(local, chain), abs=1e-9
    )


def test_a_chain_with_no_invariant_ruler_says_so(caplog):
    """A marker set that cannot measure its chain warns instead of reporting model size.

    ``fly38`` labels neither the ``neck`` nor ``abdomen0..4``, so on that skeleton
    neither chain has a single separation to measure. Returning 1.0 is the only
    available answer, but on its own it reads as a *measurement* that this fly matches
    the model -- and the overlay would then draw a confidently mis-sized head. The
    warning is what separates "the same size" from "no ruler".
    """
    import logging

    from deeperfly.inverse_kinematics.articulation import (
        estimate_chain_scale,
        load_articulation,
    )

    fly = deepfly3d_skeleton()
    index = {n: i for i, n in enumerate(fly.point_names)}
    chain = load_articulation().chain("head")
    assert chain.base_point not in index, "fly38 does not label the neck"
    rng = np.random.default_rng(7)
    pts = np.full((3, fly.n_points, 3), np.nan)
    _place_chain_markers(
        chain,
        _bent_angles(chain, rng),
        (np.eye(3), 1.0, np.zeros(3)),
        pts,
        index,
        size=1.4,
    )
    cols = [index.get(m, -1) for m in chain.marker_names]
    local = np.stack(
        [pts[:, c] if c >= 0 else np.full((3, 3), np.nan) for c in cols], axis=1
    )

    with caplog.at_level(logging.WARNING, logger="deeperfly"):
        assert estimate_chain_scale(local, chain) == 1.0
    assert "never observed" in caplog.text and "head" in caplog.text


def test_estimate_chain_scale_defaults_to_one_when_unobserved():
    """With no frame where the ruler is observed, the size stays at the model (1.0)."""
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


# -- whole-chain calibration ---------------------------------------------------


def _synthetic_chain_markers(chain, *, scale, shift, angles):
    """Markers a chain of that size, at that root, in those poses, would produce."""
    from deeperfly.inverse_kinematics.articulation import chain_markers

    return np.stack([chain_markers(chain, a, scale, shift) for a in angles])


def _plausible_poses(chain, n=24, seed=0):
    """Angle sequences inside the chain's own bounds, away from the limits."""
    lo, hi = chain.bounds
    rng = np.random.default_rng(seed)
    mid = 0.5 * (lo + hi)
    half = 0.35 * (hi - lo)
    return mid + half * (2.0 * rng.random((n, len(lo))) - 1.0)


@pytest.mark.parametrize("chain_name", ["head", "abdomen"])
@pytest.mark.parametrize("truth", [0.85, 1.0, 1.35])
def test_calibrate_chain_recovers_a_known_size(chain_name, truth):
    """On markers generated from a known size, the fit returns that size.

    Deliberately seeded from a *wrong* ruler value, because that is the case it exists
    for: the abdomen's single ruler over-reads by 10-19% and the seed must not be where
    the answer comes from.
    """
    from deeperfly.inverse_kinematics.articulation import (
        calibrate_chain,
        load_articulation,
    )

    chain = load_articulation().chain(chain_name)
    angles = _plausible_poses(chain)
    markers = _synthetic_chain_markers(
        chain, scale=truth, shift=np.zeros(3), angles=angles
    )
    got, shift = calibrate_chain(markers, chain, seed_scale=truth * 1.25, n_frames=8)
    assert got == pytest.approx(truth, abs=5e-3)
    np.testing.assert_allclose(
        shift, np.zeros(3), atol=1e-12
    )  # not fitted -> untouched


def test_calibrate_chain_recovers_a_known_size_and_root_together():
    """With the root free, a chain displaced AND resized is recovered in both.

    Size and root translation are coupled through a null direction of the fit, so a test
    that moves only one of them cannot see the coupling that this function exists to
    handle.
    """
    from deeperfly.inverse_kinematics.articulation import (
        calibrate_chain,
        load_articulation,
    )

    chain = load_articulation().chain("abdomen")
    truth_scale, truth_shift = 1.18, np.array([0.06, 0.0, -0.17])
    angles = _plausible_poses(chain, seed=3)
    markers = _synthetic_chain_markers(
        chain, scale=truth_scale, shift=truth_shift, angles=angles
    )
    scale, shift = calibrate_chain(
        markers, chain, seed_scale=1.46, fit_root=True, n_frames=8
    )
    assert scale == pytest.approx(truth_scale, abs=0.02)
    np.testing.assert_allclose(shift, truth_shift, atol=0.02)


def test_calibrate_chain_never_moves_a_midline_root_sideways():
    """A midline chain's fitted root stays in the sagittal plane.

    Its lateral joints and a sideways root shift express the *same* displacement, so
    freeing both attributes the animal's lateral bend to whichever the optimiser reaches
    first. The plane is derived from the chain, not configured -- the head, which carries
    the two antennae, is not midline and keeps all three components.
    """
    from deeperfly.inverse_kinematics.articulation import (
        _is_midline,
        calibrate_chain,
        load_articulation,
    )

    art = load_articulation()
    abdomen, head = art.chain("abdomen"), art.chain("head")
    assert _is_midline(abdomen) and not _is_midline(head)

    # markers really are displaced sideways: the fit must NOT absorb it into the root
    angles = _plausible_poses(abdomen, seed=5)
    markers = _synthetic_chain_markers(
        abdomen, scale=1.2, shift=np.array([0.0, 0.12, 0.0]), angles=angles
    )
    _, shift = calibrate_chain(
        markers, abdomen, seed_scale=1.2, fit_root=True, n_frames=6
    )
    assert shift[1] == 0.0

    _, head_shift = calibrate_chain(
        _synthetic_chain_markers(
            head,
            scale=1.2,
            shift=np.array([0.0, 0.09, 0.0]),
            angles=_plausible_poses(head, seed=6),
        ),
        head,
        seed_scale=1.2,
        fit_root=True,
        n_frames=6,
    )
    assert abs(head_shift[1]) > 0.01  # not midline -> its y IS measured


def test_calibrate_chain_keeps_the_seed_when_the_chain_is_unobserved(caplog):
    """An all-NaN chain cannot be fitted, and says so instead of returning a number."""
    from deeperfly.inverse_kinematics.articulation import (
        calibrate_chain,
        load_articulation,
    )

    chain = load_articulation().chain("abdomen")
    nan = np.full((7, len(chain.marker_names), 3), np.nan)
    # Name the logger: a CLI test elsewhere calls logging.basicConfig, so a caplog that
    # relies on the root logger passes alone and captures nothing in a full-suite run.
    with caplog.at_level("WARNING", logger="deeperfly"):
        scale, shift = calibrate_chain(nan, chain, seed_scale=1.31)
    assert scale == pytest.approx(1.31)
    np.testing.assert_allclose(shift, np.zeros(3))
    assert "never fully observed" in caplog.text


def test_chain_markers_agrees_with_chain_fk():
    """The calibration's forward model is the solver's, not a second copy of it.

    Rolling a chain's kinematics by hand is a live trap: composing the rotations in the
    opposite order is invisible on the abdomen (its axes are parallel, so they commute)
    and wrong by ~0.9 model units on the head's three-axis ball joint.
    """
    from deeperfly.inverse_kinematics.articulation import (
        chain_markers,
        load_articulation,
    )
    from deeperfly.inverse_kinematics.forward import chain_fk

    rng = np.random.default_rng(0)
    for chain in load_articulation().chains:
        anchors = np.asarray(chain.anchors, float)
        neutral = np.asarray(chain.marker_neutral, float)
        base = anchors[0]
        lo, hi = chain.bounds
        for _ in range(25):
            f = float(rng.uniform(0.7, 1.6))
            theta = lo + (hi - lo) * rng.random(len(lo))
            mine = chain_markers(chain, theta, f)
            theirs = chain_fk(
                base + f * (anchors - base),
                np.asarray(chain.axes, float),
                tuple(int(d) for d in chain.marker_depth),
                theta,
                base + f * (neutral - base),
            )
            np.testing.assert_allclose(mine, theirs, atol=1e-12)
