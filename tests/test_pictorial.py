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
    """``(V, 1, N, K, 2)`` / ``(V, 1, N, K)`` with the true projection as peak 0."""
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
    # Strongest peak at (row=8, col=40) -> normalized (x, y).
    np.testing.assert_allclose(xy[0, 0], [40 / ww, 8 / hh], atol=1e-6)
    np.testing.assert_allclose(xy[0, 1], [10 / ww, 20 / hh], atol=1e-6)


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
    assert sorted(len(c) for c in chains) == [1, 1, 3, 3, 5, 5, 5, 5, 5, 5]
    # Each leg chain is a contiguous thorax_coxa..claw run.
    legs = [c for c in chains if len(c) == 5]
    for c in legs:
        assert c == list(range(c[0], c[0] + 5))


# -- bone-length prior refactor guard ----------------------------------------


def test_bone_length_targets_matches_manual(cameras, fly, rng):
    pts3d = fly_cloud(rng)[None].repeat(4, 0)  # (F=4, N, 3)
    pts2d = np.asarray(cameras.project(pts3d))  # (V, F, N, 2)
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


def test_pictorial_recovers_decoyed_joint(cameras, fly, rng):
    pts3d = fly_cloud(rng)
    proj = np.asarray(cameras.project(pts3d))  # (V, N, 2)
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
        cameras, fly, cands, argmax, bone_max_frames=None
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
    pts3d = fly_cloud(rng)[None]  # (T=1, N, 3)
    proj = np.asarray(cameras.project(pts3d))  # (V, 1, N, 2)
    xy = np.full((*proj.shape[:3], 3, 2), np.nan)
    sc = np.zeros((*proj.shape[:3], 3))
    xy[..., 0, :] = proj
    sc[..., 0] = 0.9
    cands = pictorial.Candidates(xy=xy, score=sc)

    result = run_from_points2d(
        cameras,
        fly,
        proj[:, :, :, 0, :] if proj.ndim == 5 else proj,
        do_calibrate=False,
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
        run_from_points2d(cameras, fly, proj, do_calibrate=False, do_pictorial=True)
