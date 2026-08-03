"""Solving a camera rig from hand labels -- the from-scratch calibration.

The headline test is :func:`test_a_known_rig_is_recovered_from_a_cold_start`: build a
synthetic scene, project it through a *known* 7-camera rig, hand the 2D back with no prior
whatsoever, and check the recovered rig against the truth. That is the only check that can
tell "the residual is small" from "the geometry is right" -- an under-determined bundle
adjustment produces the first without the second, which is the failure mode this whole
module is arranged to prevent.

The comparison is up to a **similarity transform** (rotation, translation, scale), because
those seven degrees of freedom are genuinely unknowable from correspondences alone. Getting
that wrong would make a correct solve look broken.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import CAMERA_NAMES, HEIGHT, WIDTH

from deeperfly.calibration_solve import (
    MIN_EQUATION_RATIO,
    Observations,
    build_observations,
    conditioning,
    covisibility,
    initialize_extrinsics,
    solve_rig,
)
from deeperfly.cameras import CameraGroup
from deeperfly.config import Config

SIZES = {name: (HEIGHT, WIDTH) for name in CAMERA_NAMES}


@pytest.fixture
def truth() -> CameraGroup:
    """The packaged 7-camera orbit rig, used as ground truth to recover."""
    return Config.default().camera_group(image_sizes=SIZES)


def _scene(truth, *, n_static=8, n_frames=6, n_kp=12, noise=0.4, seed=0):
    """``(landmark_xy, keypoint_xy)`` for a synthetic scene seen by ``truth``.

    The two differ in exactly the way that matters: the landmarks are **static** (one 3D
    point, re-observed every frame) and spread through the scene volume; the keypoints
    **move** every frame and stay in a small blob, like an animal.
    """
    rng = np.random.default_rng(seed)
    n_views = len(truth.names)
    static3d = rng.uniform(-25, 25, size=(n_static, 3))
    animal3d = rng.uniform(-4, 4, size=(n_frames, n_kp, 3))

    landmark_xy = np.full((n_views, n_frames, n_static, 2), np.nan)
    keypoint_xy = np.full((n_views, n_frames, n_kp, 2), np.nan)
    for t in range(n_frames):
        landmark_xy[:, t] = np.asarray(truth.project(static3d)) + rng.normal(
            0, noise, (n_views, n_static, 2)
        )
        keypoint_xy[:, t] = np.asarray(truth.project(animal3d[t])) + rng.normal(
            0, noise, (n_views, n_kp, 2)
        )
    return landmark_xy, keypoint_xy


def _observations(truth, use, *, frames=None, **kwargs):
    landmark_xy, keypoint_xy = _scene(truth, **kwargs)
    return build_observations(
        view_names=list(truth.names),
        landmark_xy=landmark_xy,
        landmark_names=[f"lm{i}" for i in range(landmark_xy.shape[2])],
        landmark_static=np.ones(landmark_xy.shape[2], dtype=bool),
        keypoint_xy=keypoint_xy,
        keypoint_names=[f"kp{i}" for i in range(keypoint_xy.shape[2])],
        frames=frames,
        use=use,
    )


def _centre_error_pct(truth: CameraGroup, solved: CameraGroup) -> float:
    """Max camera-centre error after the best similarity alignment, as % of rig radius.

    Umeyama alignment, because rotation, translation and scale are exactly the gauge
    freedoms the problem cannot determine. Comparing raw coordinates would fail a perfect
    solve.
    """
    names = list(truth.names)
    a = np.stack([truth[n].position for n in names])
    b = np.stack([solved[n].position for n in names])
    ac, bc = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(bc.T @ ac)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    scale = s.sum() / (bc**2).sum()
    err = np.linalg.norm(scale * (bc @ rot) - ac, axis=1)
    return float(100 * err.max() / np.linalg.norm(ac, axis=1).mean())


# -- the headline check --------------------------------------------------------


@pytest.mark.parametrize("use", ["landmarks", "keypoints", "both"])
def test_a_known_rig_is_recovered_from_a_cold_start(truth, use):
    """No prior at all: essential matrix -> PnP -> bundle adjustment -> the true rig.

    Recovering the camera centres to well under 1% of the rig radius is what separates a
    solve that works from one that merely reports a small residual.
    """
    obs = _observations(truth, use)
    cond = conditioning(obs)
    assert cond["ok"], cond["reasons"]

    rvecs, tvecs, init = initialize_extrinsics(obs, truth.intrs, truth.dists)
    assert init["failed"] == [], f"views left unregistered: {init['failed']}"
    assert np.isfinite(rvecs).all() and np.isfinite(tvecs).all()

    result = solve_rig(
        obs,
        intrinsics=truth.intrs,
        dists=truth.dists,
        rvecs=rvecs,
        tvecs=tvecs,
        cold_start=True,
    )
    assert result.ok
    assert result.quality["rms_reproj_px"] < 2.0
    assert _centre_error_pct(truth, result.cameras) < 1.0


def test_the_orbit_prior_path_needs_no_sfm(truth):
    """An existing rig initializes from its own config, which must keep working.

    This is the continuity guarantee: a from-scratch code path must not change what
    today's rigs do.
    """
    obs = _observations(truth, "both")
    result = solve_rig(
        obs,
        intrinsics=truth.intrs,
        dists=truth.dists,
        rvecs=truth.rvecs,
        tvecs=truth.tvecs,
    )
    assert result.ok
    assert result.quality["rms_reproj_px"] < 1.0
    assert _centre_error_pct(truth, result.cameras) < 0.5


def test_a_free_camera_at_the_rodrigues_singularity_does_not_poison_the_solve(truth):
    """rvec == 0 exactly is a hole in the axis-angle parameterization, not a small angle.

    ``dR/drvec`` carries ``sin(theta)/theta``, which autodiff evaluates as 0/0 at zero --
    so a *free* camera sitting there contributes a NaN Jacobian column and the first
    trust-region step dies. A cold-start SfM initialization puts its reference view exactly
    there, which is why this is guarded rather than left to chance.
    """
    obs = _observations(truth, "both")
    rvecs = np.asarray(truth.rvecs, dtype=float).copy()
    tvecs = np.asarray(truth.tvecs, dtype=float).copy()
    rvecs[3] = 0.0  # a FREE view (view 0 is the fixed one) parked on the singularity

    result = solve_rig(
        obs, intrinsics=truth.intrs, dists=truth.dists, rvecs=rvecs, tvecs=tvecs
    )
    assert np.isfinite(result.cameras.rvecs).all()
    assert np.isfinite(result.quality["rms_reproj_px"])


# -- assembling observations ---------------------------------------------------


def test_a_static_landmark_is_one_track_however_many_frames_see_it(truth):
    """This is the whole reason landmarks exist: many observations, three unknowns."""
    obs = _observations(truth, "landmarks", n_static=5, n_frames=10)
    assert obs.n_tracks == 5
    assert all(t.static for t in obs.tracks)
    # Ten frames x seven views collapsed onto seven averaged observations per track.
    assert obs.n_observations == 5 * len(truth.names)
    assert all(t.n_observations == 10 * len(truth.names) for t in obs.tracks)


def test_a_moving_keypoint_is_a_separate_track_per_frame(truth):
    """The animal moved, so frame t and frame t+1 are different 3D points."""
    obs = _observations(truth, "keypoints", n_frames=4, n_kp=3)
    assert obs.n_tracks == 12
    assert not any(t.static for t in obs.tracks)


def test_a_static_landmarks_scatter_is_measured(truth):
    """Averaging a static landmark's frames is free; the scatter it reveals is the point.

    A landmark whose pixel wanders is not static -- or was labeled on a different speck in
    a different frame -- and nothing else in the pipeline would say so.
    """
    landmark_xy, _ = _scene(truth, n_static=2, n_frames=8, noise=0.0)
    # Make landmark 1 drift steadily; landmark 0 stays put.
    landmark_xy[:, :, 1, 0] += np.arange(8)[None, :] * 3.0

    obs = build_observations(
        view_names=list(truth.names),
        landmark_xy=landmark_xy,
        landmark_names=["still", "drifting"],
        landmark_static=np.ones(2, dtype=bool),
        use="landmarks",
    )
    scatter = {t.label: t.scatter_px for t in obs.tracks}
    assert scatter["still"] < 0.01
    assert scatter["drifting"] > 1.0


def test_only_the_named_frames_are_used(truth):
    """The caller restricts to reviewed frames; a half-labeled one biases a track."""
    obs = _observations(truth, "keypoints", n_frames=6, n_kp=2, frames=[1, 3])
    assert {t.frame for t in obs.tracks} == {1, 3}


def test_a_track_seen_by_one_view_is_dropped(truth):
    """It would add three unknowns and constrain nothing."""
    _, keypoint_xy = _scene(truth, n_frames=1, n_kp=3)
    keypoint_xy[1:, 0, 0] = np.nan  # point 0 visible in one view only

    obs = build_observations(
        view_names=list(truth.names),
        keypoint_xy=keypoint_xy,
        keypoint_names=["a", "b", "c"],
        use="keypoints",
    )
    assert [t.label for t in obs.tracks] == ["b@0", "c@0"]


def test_an_unknown_use_is_refused(truth):
    with pytest.raises(ValueError, match="use must be"):
        _observations(truth, "vibes")


def test_use_selects_which_tracks_drive_the_solve(truth):
    """The operator's choice: skeleton points, landmarks, or both."""
    lm_only = _observations(truth, "landmarks", n_static=4, n_frames=2, n_kp=5)
    kp_only = _observations(truth, "keypoints", n_static=4, n_frames=2, n_kp=5)
    both = _observations(truth, "both", n_static=4, n_frames=2, n_kp=5)
    assert lm_only.n_tracks == 4
    assert kp_only.n_tracks == 10
    assert both.n_tracks == 14


# -- the conditioning gate -----------------------------------------------------


def test_a_disconnected_covisibility_graph_is_refused(truth):
    """Two half-rigs sharing no points have an unknowable relative pose.

    A camera ring is the realistic case: left and right views may share nothing, with a
    single front view as the only bridge. The gate has to name the groups, because "it
    failed" is not actionable and "{rh,rm,rf} vs {lf,lm,lh}" is.
    """
    _, keypoint_xy = _scene(truth, n_frames=2, n_kp=6)
    # Views 0-2 see the first three points, views 4-6 the last three, view 3 nothing.
    keypoint_xy[0:3, :, 3:] = np.nan
    keypoint_xy[4:7, :, :3] = np.nan
    keypoint_xy[3] = np.nan

    obs = build_observations(
        view_names=list(truth.names),
        keypoint_xy=keypoint_xy,
        keypoint_names=[f"p{i}" for i in range(6)],
        use="keypoints",
    )
    cond = conditioning(obs)
    assert not cond["ok"]
    joined = " ".join(cond["reasons"])
    assert "disconnected" in joined
    assert "observe no tracks" in joined  # view 3 named separately
    assert len(cond["components"]) == 3  # two groups plus the blind view


def test_too_few_observations_is_refused_with_the_ratio(truth):
    obs = _observations(truth, "keypoints", n_frames=1, n_kp=2)
    cond = conditioning(obs)
    assert not cond["ok"]
    assert f"{MIN_EQUATION_RATIO}x" in " ".join(cond["reasons"])
    assert cond["ratio"] < MIN_EQUATION_RATIO


def test_freeing_intrinsics_raises_the_bar(truth):
    """Each freed parameter is another unknown, so the gate must account for it."""
    obs = _observations(truth, "both")
    tight = conditioning(obs)
    loose = conditioning(obs, free_focal=True, free_k1=True)
    assert loose["unknowns"] > tight["unknowns"]
    assert loose["ratio"] < tight["ratio"]


def test_a_healthy_set_passes_and_reports_the_weakest_pair(truth):
    obs = _observations(truth, "both")
    cond = conditioning(obs)
    assert cond["ok"] and cond["reasons"] == []
    assert len(cond["components"]) == 1
    assert cond["weakest_pair"]["shared"] > 0
    assert set(cond["per_view_tracks"]) == set(truth.names)


def test_covisibility_counts_shared_tracks(truth):
    obs = _observations(truth, "landmarks", n_static=5, n_frames=1)
    co = covisibility(obs)
    assert co.shape == (7, 7)
    assert co[0, 0] == 5  # the diagonal is each view's own count
    assert co[0, 1] == 5  # every view sees every landmark here


def test_conditioning_of_nothing_is_refused_not_crashed():
    """An operator who has labeled nothing yet must get the readiness list, not a stack."""
    obs = Observations(np.zeros((3, 0, 2)), [], ["a", "b", "c"])
    cond = conditioning(obs)
    assert not cond["ok"]
    assert cond["reasons"]


# -- scale --------------------------------------------------------------------


def test_a_known_distance_pins_the_scale(truth):
    """The existing bone-length prior IS a scale bar -- no new solver code.

    With a metric distance between two tracks supplied, the recovered rig should come out
    at the true scale rather than an arbitrary one.
    """
    obs = _observations(truth, "landmarks", n_static=8, n_frames=4, noise=0.1)
    # Triangulate with the truth to learn the real distance between two tracks.
    exact = np.asarray(truth.triangulate(obs.pts2d))
    distance = float(np.linalg.norm(exact[0] - exact[1]))

    result = solve_rig(
        obs,
        intrinsics=truth.intrs,
        dists=truth.dists,
        rvecs=truth.rvecs,
        tvecs=truth.tvecs,
        scale_pair=(0, 1),
        scale_distance=distance,
    )
    assert result.quality["gauge"]["scale_fixed_by"] == "known_distance"
    got = float(np.linalg.norm(result.pts3d[0] - result.pts3d[1]))
    assert abs(got - distance) / distance < 0.02


def test_with_no_scale_reference_the_gauge_says_none(truth):
    obs = _observations(truth, "landmarks", n_frames=2)
    result = solve_rig(
        obs,
        intrinsics=truth.intrs,
        dists=truth.dists,
        rvecs=truth.rvecs,
        tvecs=truth.tvecs,
    )
    assert result.quality["gauge"]["scale_fixed_by"] == "none"


# -- the report ---------------------------------------------------------------


def test_the_report_names_the_worst_tracks(truth):
    """A bad residual has to be navigable, so every row carries its label."""
    obs = _observations(truth, "both", n_frames=3)
    result = solve_rig(
        obs,
        intrinsics=truth.intrs,
        dists=truth.dists,
        rvecs=truth.rvecs,
        tvecs=truth.tvecs,
    )
    rows = result.report["per_track"]
    assert rows and all("label" in r and "max_reproj_px" in r for r in rows)
    # Sorted worst-first, and capped (a silently truncated list reads as "all of them").
    assert rows == sorted(rows, key=lambda r: -r["max_reproj_px"])
    assert len(rows) <= 20


def test_the_quality_block_is_per_camera(truth):
    obs = _observations(truth, "both", n_frames=3)
    result = solve_rig(
        obs,
        intrinsics=truth.intrs,
        dists=truth.dists,
        rvecs=truth.rvecs,
        tvecs=truth.tvecs,
    )
    assert set(result.quality["per_camera_rms_px"]) == set(truth.names)
    assert result.report["observations"]["views"] == 7
