"""Tests for the pictorial-structures (PS) 2D->3D corrector.

The detector is stubbed out: we project a known 3D fly through the synthetic test
rig to get ground-truth 2D, then build candidate peak sets (optionally with a
wrong arg-max "decoy" plus the true location as a secondary peak) and check that
PS recovers the joint where the default triangulation path can only veto it.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import fly_masked

from deeperfly import pictorial
from deeperfly.pipeline import _bone_prior, reconstruct, run_from_points2d
from deeperfly.results import PoseResult


def fly_cloud(rng, n_pts=38):
    """A small static 3D fly cloud near the world origin."""
    return rng.uniform(-1.2, 1.2, size=(n_pts, 3))


def candidates_from_proj(proj, k, *, score=0.9):
    """``(V, 1, P, K, 2)`` / ``(V, 1, P, K)`` with the true projection as peak 0."""
    v, n, _ = proj.shape
    xy = np.full((v, 1, n, k, 2), np.nan)
    sc = np.zeros((v, 1, n, k))
    xy[:, 0, :, 0] = proj
    sc[:, 0, :, 0] = score
    return xy, sc


# -- peak extraction ---------------------------------------------------------


def test_peak_candidates_finds_ordered_bumps():
    hh, ww = 32, 64
    hm = np.zeros((1, hh, ww))
    yy, xx = np.mgrid[0:hh, 0:ww]
    # Two bumps; the (8, 40) one is stronger so must come first.
    hm[0] += 1.0 * np.exp(-((yy - 20) ** 2 + (xx - 10) ** 2) / 4.0)
    hm[0] += 1.5 * np.exp(-((yy - 8) ** 2 + (xx - 40) ** 2) / 4.0)
    xy, score = pictorial.peak_candidates(hm, k=2, radius=2)
    assert score[0, 0] > score[0, 1]  # ordered by strength
    # Strongest peak at (row=8, col=40) -> normalized (x, y), cell-centre (+0.5).
    np.testing.assert_allclose(xy[0, 0], [(40 + 0.5) / ww, (8 + 0.5) / hh], atol=1e-6)
    np.testing.assert_allclose(xy[0, 1], [(10 + 0.5) / ww, (20 + 0.5) / hh], atol=1e-6)


def test_peak_candidates_pads_when_too_few():
    hm = np.zeros((1, 16, 16))
    hm[0, 5, 5] = 1.0  # a single peak
    xy, score = pictorial.peak_candidates(hm, k=4)
    assert np.isfinite(xy[0, 0]).all() and score[0, 0] == 1.0
    assert np.isnan(xy[0, 1:]).all() and (score[0, 1:] == 0).all()


# -- skeleton chains ---------------------------------------------------------


def test_skeleton_chains_partition_fly(fly):
    chains = pictorial.skeleton_chains(fly)
    covered = sorted(j for c in chains for j in c)
    assert covered == list(range(fly.n_points))  # exact partition, no dupes
    # fly38b: six 5-point legs, a 5-point midline abdomen, two antennae and the neck.
    assert sorted(len(c) for c in chains) == [1, 1, 1, 5, 5, 5, 5, 5, 5, 5]
    # Each leg chain is a contiguous thorax_coxa..claw run.
    legs = [c for c in chains if len(c) == 5]
    for c in legs:
        assert c == list(range(c[0], c[0] + 5))


# -- bone-length prior refactor guard ----------------------------------------


def test_bone_length_targets_matches_manual(cameras, fly, rng):
    pts3d = fly_cloud(rng)[None].repeat(4, 0)  # (F=4, P, 3)
    pts2d = np.asarray(cameras.project(pts3d))  # (V, F, P, 2)
    i, j, targets = pictorial.bone_length_targets(cameras, pts2d, fly)
    # The true bone lengths, recovered exactly from clean multi-view geometry.
    expect = np.linalg.norm(pts3d[0, i] - pts3d[0, j], axis=-1)
    np.testing.assert_allclose(targets, expect, atol=1e-6)


def test_bone_prior_uses_shared_targets(cameras, fly, rng):
    """`_bone_prior` must tile the shared per-bone targets across frames."""
    pts3d = fly_cloud(rng)[None].repeat(3, 0)
    pts2d = np.asarray(cameras.project(pts3d))
    pairs, tiled = _bone_prior(cameras, pts2d, fly)
    _, _, targets = pictorial.bone_length_targets(cameras, pts2d, fly)
    n_frames, n_bones = 3, len(targets)
    assert pairs.shape == (n_frames * n_bones, 2)
    np.testing.assert_allclose(tiled, np.tile(targets, n_frames))


# -- recovery vs rejection (headline) ----------------------------------------


def test_pictorial_recovers_decoyed_joint(cameras, deepfly3d, rng):
    pts3d = fly_cloud(rng)
    proj = np.asarray(cameras.project(pts3d))  # (V, P, 2)
    k = 5
    xy, sc = candidates_from_proj(proj, k)

    # Joint 21 (right leg, seen by several right cameras): in camera 0 the arg-max
    # is a 40 px decoy while the *true* location is only the secondary peak.
    decoy_view, joint = 0, 21
    xy[decoy_view, 0, joint, 1] = proj[decoy_view, joint]  # true as secondary
    sc[decoy_view, 0, joint, 1] = 0.9
    xy[decoy_view, 0, joint, 0] = proj[decoy_view, joint] + [40.0, 40.0]  # decoy
    sc[decoy_view, 0, joint, 0] = 0.95

    cands = pictorial.Candidates(xy=xy, score=sc)
    argmax = xy[:, :, :, 0, :]  # the (wrong) single-peak detections

    ps3d, _, _ = pictorial.reconstruct(
        cameras, deepfly3d, cands, argmax, bone_max_frames=None
    )
    # The greedy path triangulates the arg-max (including the decoy).
    rp3d, _, _ = reconstruct(cameras, fly_masked(argmax))

    ps_err = np.linalg.norm(ps3d[0, joint] - pts3d[joint])
    rp_err = np.linalg.norm(rp3d[0, joint] - pts3d[joint])
    assert ps_err < 1e-3  # PS recovers the true 3D from the secondary peak
    assert rp_err > 10 * ps_err  # the greedy fit is dragged off by the decoy


def test_pictorial_clean_matches_truth(cameras, fly, rng):
    """With only the true peak (K=1) PS reconstructs the visible joints exactly."""
    pts3d = fly_cloud(rng)
    proj = np.asarray(cameras.project(pts3d))
    xy, sc = candidates_from_proj(proj, k=1)
    cands = pictorial.Candidates(xy=xy, score=sc)
    ps3d, _, _ = pictorial.reconstruct(
        cameras, fly, cands, xy[:, :, :, 0, :], bone_max_frames=None
    )
    seen = np.isfinite(ps3d[0]).all(-1)
    assert seen.sum() >= 30  # most joints are multi-view visible
    np.testing.assert_allclose(ps3d[0, seen], pts3d[seen], atol=1e-4)


# -- chain DP ----------------------------------------------------------------


def test_chain_dp_prefers_anatomical_bone_length():
    p0 = np.array([[0.0, 0.0, 0.0]])  # joint 0: single hypothesis
    # joint 1: a tempting (lower-cost) wrong-length hypothesis vs the correct one.
    good = np.array([1.0, 0.0, 0.0])  # length 1.0 from p0 (== target)
    bad = np.array([3.0, 0.0, 0.0])  # length 3.0 (anatomically wrong)
    pos = {0: p0, 1: np.stack([good, bad])}
    unary = {0: np.array([0.0]), 1: np.array([0.0, -1.0])}  # 'bad' has more evidence
    target_map = {(0, 1): 1.0}

    no_prior = pictorial._chain_dp(
        [0, 1], pos, unary, target_map, lam=0.0, scale=1.0, huber=0.5
    )
    with_prior = pictorial._chain_dp(
        [0, 1], pos, unary, target_map, lam=50.0, scale=1.0, huber=0.5
    )
    assert no_prior[1] == 1  # evidence alone -> the wrong (stronger) candidate
    assert with_prior[1] == 0  # bone prior overrides it -> anatomically correct


def test_chain_dp_skips_jointless_gaps():
    pos = {0: np.zeros((1, 3)), 1: np.empty((0, 3)), 2: np.ones((1, 3))}
    unary = {0: np.array([0.0]), 1: np.array([]), 2: np.array([0.0])}
    choice = pictorial._chain_dp(
        [0, 1, 2], pos, unary, {}, lam=1.0, scale=1.0, huber=0.5
    )
    assert choice == {0: 0, 2: 0}  # joint 1 (no hypotheses) simply omitted


# -- temporal term -----------------------------------------------------------


def test_temporal_term_suppresses_jump(cameras, fly, rng):
    pts3d = fly_cloud(rng)
    joint = 15  # r_antenna: a singleton (no bone coupling) seen by several cameras
    proj = np.asarray(cameras.project(pts3d))

    near = pts3d[joint]
    far = near + np.array([0.6, -0.4, 0.5])  # a large 3D displacement
    proj_far = np.asarray(cameras.project(far[None]))[:, 0]  # (V, 2)

    v, n = proj.shape[0], fly.n_points
    k = 2
    xy = np.full((v, n, k, 2), np.nan)
    sc = np.zeros((v, n, k))
    # `far` supported by 3 views at high score; `near` by 3 views at lower score.
    for view in (0, 1, 2):
        xy[view, joint, 0] = proj_far[view]
        sc[view, joint, 0] = 0.95
    for view in (0, 1, 3):
        xy[view, joint, 1] = proj[view, joint]
        sc[view, joint, 1] = 0.9

    chains = pictorial.skeleton_chains(fly)
    common = dict(target_map={}, chains=chains, scale=1.0, inlier_px=5.0)

    x_no_t, _ = pictorial.solve_frame(
        cameras, fly, xy, sc, mu=0.0, prev_pts3d=None, **common
    )
    x_temporal, _ = pictorial.solve_frame(
        cameras, fly, xy, sc, mu=50.0, prev_pts3d=pts3d, **common
    )
    assert np.linalg.norm(x_no_t[joint] - far) < 1e-3  # evidence -> the far jump
    assert np.linalg.norm(x_temporal[joint] - near) < 1e-3  # temporal -> stays near


# -- degenerate fallback -----------------------------------------------------


def test_single_view_joint_is_nan(cameras, fly, rng):
    pts3d = fly_cloud(rng)
    proj = np.asarray(cameras.project(pts3d))
    xy, sc = candidates_from_proj(proj, k=2)
    # Strip joint 4 down to a single camera -> impossible to triangulate.
    lonely = 4
    xy[1:, 0, lonely] = np.nan
    sc[1:, 0, lonely] = 0.0
    cands = pictorial.Candidates(xy=xy, score=sc)
    ps3d, _, _ = pictorial.reconstruct(
        cameras, fly, cands, xy[:, :, :, 0, :], bone_max_frames=None
    )
    assert np.isnan(ps3d[0, lonely]).all()
    assert np.isfinite(ps3d[0, 2]).all()  # neighbors unaffected


# -- pipeline integration ----------------------------------------------------


def test_run_from_points2d_pictorial(cameras, fly, rng):
    pts3d = fly_cloud(rng)[None]  # (T=1, P, 3)
    proj = np.asarray(cameras.project(pts3d))  # (V, 1, P, 2)
    xy = np.full((*proj.shape[:3], 3, 2), np.nan)
    sc = np.zeros((*proj.shape[:3], 3))
    xy[..., 0, :] = proj
    sc[..., 0] = 0.9
    cands = pictorial.Candidates(xy=xy, score=sc)

    result = run_from_points2d(
        cameras,
        fly,
        proj[:, :, :, 0, :] if proj.ndim == 5 else proj,
        do_bundle_adjust=False,
        do_pictorial=True,
        candidates=cands,
    )
    assert isinstance(result, PoseResult)
    assert result.meta["pictorial"] is True
    assert result.skeleton.n_points == 38
    assert result.pts3d.shape == (1, 38, 3)


def test_pictorial_requires_candidates(cameras, fly, rng):
    proj = np.asarray(cameras.project(fly_cloud(rng)[None]))
    with pytest.raises(ValueError, match="requires candidates"):
        run_from_points2d(cameras, fly, proj, do_bundle_adjust=False, do_pictorial=True)


# -- election, and the arg-max fallback --------------------------------------
#
# `elect_frame` is the cheap research baseline, deliberately NOT reachable from a config:
# measured on held-out animals it swings 14.0-68.8% of the available gain where the
# shipped decoder holds 53.3-66.2%, so it is the higher-variance estimator, not a cheaper
# equivalent. `k` is the production accuracy/cost dial.


def test_elect_frame_recovers_a_decoyed_joint(cameras, fly, rng):
    """Election finds the truth at rank 1 when rank 0 is a decoy."""
    pts3d = fly_cloud(rng)
    proj = np.asarray(cameras.project(pts3d))
    xy, sc = candidates_from_proj(proj, k=3)
    xy[:, 0, :, 1] = proj
    sc[:, 0, :, 1] = 0.7
    xy[:, 0, :, 0] = proj + rng.normal(0.0, 0.4, size=proj.shape)
    sc[:, 0, :, 0] = 0.9

    got = pictorial.elect_frame(cameras, xy[:, 0], sc[:, 0])
    argmax_err = np.linalg.norm(xy[:, 0, :, 0] - proj, axis=-1)
    got_err = np.linalg.norm(got - proj, axis=-1)
    assert got_err.mean() < argmax_err.mean(), "election did not beat the arg-max"


def test_elect_frame_never_abstains(cameras, fly, rng):
    """Unlike recovery, election always returns a real detected peak."""
    pts3d = fly_cloud(rng)
    proj = np.asarray(cameras.project(pts3d))
    xy, sc = candidates_from_proj(proj, k=2)
    xy[:, 0, :, 1] = proj + rng.normal(0.0, 20.0, size=proj.shape)  # a useless 2nd peak
    sc[:, 0, :, 1] = 0.1
    got = pictorial.elect_frame(cameras, xy[:, 0], sc[:, 0])
    assert np.isfinite(got).all(), "election must never write NaN over a finite arg-max"


def test_fallback_argmax_fills_abstentions(cameras, fly, rng):
    """With the fallback on, a cell recovery declined keeps its arg-max, not a NaN."""
    pts3d = fly_cloud(rng)
    proj = np.asarray(cameras.project(pts3d))
    xy, sc = candidates_from_proj(proj, k=1)
    # Make joint 0 unsolvable: scatter its only candidate per view so far apart that no
    # hypothesis finds cross-view support.
    xy[:, 0, 0, 0] = proj[:, 0] + rng.normal(0.0, 500.0, size=(proj.shape[0], 2))
    cands = pictorial.Candidates(xy=xy, score=sc)
    argmax2d = xy[:, :, :, 0, :]

    _, off, _ = pictorial.reconstruct(
        cameras, fly, cands, argmax2d, fallback_argmax=False, bone_max_frames=None
    )
    _, on, _ = pictorial.reconstruct(
        cameras, fly, cands, argmax2d, fallback_argmax=True, bone_max_frames=None
    )
    gap = ~np.isfinite(off).all(-1) & np.isfinite(argmax2d).all(-1)
    assert gap.any(), "no abstention to fall back from -- the test would be vacuous"
    np.testing.assert_array_equal(on[gap], argmax2d[gap])
    keep = np.isfinite(off).all(-1)
    np.testing.assert_array_equal(on[keep], off[keep])


def test_k_is_the_only_accuracy_cost_knob():
    """The config exposes one dial, and it is the one that means accuracy vs cost.

    Asserted because two richer surfaces were measured and rejected: a ``mode`` naming a
    second estimator (election is the opposite corner of a pool x commitment 2x2, so no
    single parameter honestly selects it) and a ``fallback_argmax`` switch (filling an
    abstention is 3D-neutral here, so there is nothing to trade).
    """
    from deeperfly.config import Config

    ps = Config.default().pictorial
    assert ps.k == 5
    assert not hasattr(ps, "mode"), "there is no second decoder to select"
    assert not hasattr(ps, "fallback_argmax"), "filling abstentions is not a choice"

    data = Config.default().data
    data["pictorial_structures"] = {"k": 3}
    assert Config.from_dict(data).pictorial.k == 3


def test_the_stage_always_fills_abstentions(cameras, fly, rng):
    """The pipeline never stores a 2D layer sparser than the detector it corrected."""
    from deeperfly.config import Config
    from deeperfly.pipeline import stages

    pts3d = fly_cloud(rng)
    proj = np.asarray(cameras.project(pts3d))
    xy, sc = candidates_from_proj(proj, k=1)
    xy[:, 0, 0, 0] = proj[:, 0] + rng.normal(0.0, 500.0, size=(proj.shape[0], 2))
    argmax2d = xy[:, :, :, 0, :]

    got2d, _, _ = stages.stage_pictorial_structures(
        Config.default(),
        cameras,
        fly,
        pictorial.Candidates(xy=xy, score=sc),
        argmax2d,
    )
    assert np.isfinite(got2d).all() == np.isfinite(argmax2d).all()
    finite_in = np.isfinite(argmax2d).all(-1)
    assert np.isfinite(got2d[finite_in]).all(), "the stage dropped a detected point"


# -- the padded field's candidate decode -------------------------------------------------


def test_peak_candidates_honors_a_models_own_cell_geometry():
    """A PADDED field must be placed by the model's transform, not the shared convention.

    The regression this pins is silent and large: the shared ``(c + 0.5) / W_field``
    convention normalizes by the field, which on a padded model is the PADDED input and
    not the reported frame. On the r28 multiview transformer (48 px margin, 88x152 field,
    256x512 reported frame) that misplaces the outermost candidate by ~46 model px --
    three times the 15 px a candidate is allowed to sit from a hypothesis, so every edge
    candidate would be dropped while looking like a clean run.
    """
    # A field of 12x20 stride-4 cells over a 32x64 reported frame padded by 8 px a side:
    # 20 * 4 == 64 + 2 * 8. The peak sits at the far corner, where the two conventions
    # disagree most -- and where a candidate is worth having.
    hm = np.zeros((1, 12, 20))
    hm[0, 10, 18] = 1.0

    shared, _ = pictorial.peak_candidates(hm, k=1, radius=1)
    # (18 + 0.5) / 20, (10 + 0.5) / 12 -- normalized by the FIELD's own extent, which on a
    # padded model is the padded input and not the frame the coordinates claim to be in.
    assert shared[0, 0] == pytest.approx([18.5 / 20, 10.5 / 12])

    def cells_to_normalized(cells):
        return np.stack(
            [(cells[..., 0] * 4 - 8) / 64.0, (cells[..., 1] * 4 - 8) / 32.0], axis=-1
        )

    owned, _ = pictorial.peak_candidates(
        hm, k=1, radius=1, normalize=cells_to_normalized
    )
    # Exactly 1.0 on both axes: the peak is on the reported frame's far edge, which the
    # shared convention places at 0.925 -- 4.8 px in, on a 64 px frame.
    assert owned[0, 0] == pytest.approx([1.0, 1.0])
    # The two really do disagree, and by much more than a rounding difference.
    assert abs(owned[0, 0, 0] - shared[0, 0, 0]) > 0.07


def test_peak_candidates_relative_threshold_is_scale_free():
    """The absolute gate is a claim about one detector's output scale; the relative is not.

    Measured on the r28 multiview transformer, whose field peaks near 0.08: at the shipped
    absolute 0.05 **no cell has a second candidate at all**, so recovery could only ever
    return its own input. The relative gate judges each channel against its own peak, so
    the same fraction survives whatever the field's scale.
    """
    hm = np.zeros((1, 9, 9))
    hm[0, 2, 2] = 0.08  # a multiview-transformer-scale primary peak
    hm[0, 6, 6] = 0.03  # a genuine secondary mode, 38% of it

    absolute, _ = pictorial.peak_candidates(hm, k=3, radius=1, threshold=0.05)
    assert np.isfinite(absolute[0, :, 0]).sum() == 1  # the secondary is gated away

    relative, _ = pictorial.peak_candidates(
        hm, k=3, radius=1, threshold=0.0, threshold_rel=0.2
    )
    assert np.isfinite(relative[0, :, 0]).sum() == 2

    # Scale-free: the same field at 10x reports the same two candidates.
    scaled, _ = pictorial.peak_candidates(
        hm * 10.0, k=3, radius=1, threshold=0.0, threshold_rel=0.2
    )
    assert np.isfinite(scaled[0, :, 0]).sum() == 2
    assert scaled[0, :2] == pytest.approx(relative[0, :2])
