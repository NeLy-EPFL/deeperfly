"""End-to-end pipeline tests using synthetic (stubbed) 2D detections.

The 2D detector is the only learned component; here we replace it with the
ground-truth projection of a known moving fly, optionally corrupted with noise
and gross outliers, so the whole geometry/correction pipeline can be validated
deterministically without any weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.cameras import CameraGroup
from deeperfly.io import PoseResult
from deeperfly.pipeline import calibrate, reconstruct, run_from_points2d
from deeperfly.skeleton import Skeleton
from helpers import small_rotation


@pytest.fixture
def cameras(rig) -> CameraGroup:
    return CameraGroup.from_arrays(
        rig["names"], rig["rvecs"], rig["tvecs"], rig["intrs"], rig["dists"]
    )


@pytest.fixture
def fly() -> Skeleton:
    return Skeleton.fly()


def fly_motion(rng, n_frames=12, n_pts=38):
    """A small, slowly moving 3D fly cloud near the world origin."""
    base = rng.uniform(-1.5, 1.5, size=(n_pts, 3))
    t = np.linspace(0, 1, n_frames)[:, None, None]
    wiggle = 0.2 * np.sin(2 * np.pi * (t + np.arange(n_pts)[None, :, None] / n_pts))
    return base[None] + wiggle  # (T, N, 3)


# -- reconstruct -------------------------------------------------------------


def test_reconstruct_rejects_outliers(cameras, rng):
    pts3d = fly_motion(rng)
    pts2d = np.array(cameras.project(pts3d))  # (V, T, N, 2)
    # Inject gross outliers into single views (each point still seen elsewhere).
    pts2d[1, 0, 5] += [200.0, -150.0]
    pts2d[4, 3, 20] += [-300.0, 120.0]

    recovered, cleaned, err = reconstruct(cameras, pts2d, max_drops=3)
    assert recovered.shape == pts3d.shape
    assert np.nanmax(err) < 40.0  # all gross outliers rejected
    assert not np.isnan(recovered).any()  # every point still has >= 2 good views
    np.testing.assert_allclose(recovered, pts3d, atol=1e-6)
    assert np.isnan(cleaned[1, 0, 5]).all()  # outlier observation dropped


def test_reconstruct_noisy_is_approximate(cameras, rng):
    pts3d = fly_motion(rng)
    pts2d = np.array(cameras.project(pts3d))
    pts2d += rng.normal(scale=0.2, size=pts2d.shape)  # sub-pixel detector noise
    recovered, _, _ = reconstruct(cameras, pts2d, max_drops=0)
    np.testing.assert_allclose(recovered, pts3d, atol=0.05)  # mm-scale


# -- calibrate ---------------------------------------------------------------


def perturbed_cameras(rig, *, keep=("f",)):
    """Cameras perturbed off the truth, leaving the gauge anchors at truth."""
    from deeperfly import geometry as geom

    rvecs = rig["rvecs"].copy()
    tvecs = rig["tvecs"].copy()
    for i, name in enumerate(rig["names"]):
        if name in keep:
            continue
        rmat = np.asarray(geom.rvec_to_rmat(rig["rvecs"][i]))
        rvecs[i] = np.asarray(geom.rmat_to_rvec(small_rotation(0.01, i) @ rmat))
        tvecs[i, :2] += np.random.default_rng(i).normal(scale=0.5, size=2)
    rm = rig["names"].index("rm")
    tvecs[rm, 2] = rig["tvecs"][rm, 2]  # keep the scale-fixing component at truth
    return CameraGroup.from_arrays(
        rig["names"], rvecs, tvecs, rig["intrs"], rig["dists"]
    )


def test_calibrate_recovers_perturbed_rig(rig, cameras, fly, rng):
    pts3d = fly_motion(rng, n_frames=20)
    pts2d = np.array(cameras.project(pts3d))  # ground-truth observations
    cams0 = perturbed_cameras(rig)

    opt, result = calibrate(
        cams0,
        pts2d,
        skeleton=fly,
        fixed=["*.intr", "f.rvec", "f.tvec", "rm.tvec[2]"],
        bone_prior=False,
        loss="linear",
        max_nfev=2000,
    )
    # Refined cameras reproject the clean observations almost exactly.
    proj = np.asarray(opt.project(pts3d))
    assert np.nanmax(np.abs(proj - pts2d)) < 1e-2
    assert result.cost < 1e-4


# -- full pipeline -----------------------------------------------------------


def test_run_from_points2d_end_to_end(cameras, fly, rng):
    pts3d = fly_motion(rng, n_frames=16)
    pts2d = np.array(cameras.project(pts3d))
    pts2d += rng.normal(scale=0.1, size=pts2d.shape)
    pts2d[2, 5, 9] += [250.0, 250.0]  # an outlier
    conf = np.ones(pts2d.shape[:3])

    result = run_from_points2d(
        cameras,
        fly,
        pts2d,
        conf,
        do_calibrate=False,
        max_drops=3,
        smooth="one_euro",
        fps=100.0,
        smooth_kwargs={"mincutoff": 0.5},
        meta={"source": "synthetic"},
    )
    assert isinstance(result, PoseResult)
    assert result.pts3d.shape == pts3d.shape
    assert result.pts3d_smoothed is not None
    assert not np.isnan(result.pts3d).any()  # every fly point recoverable
    assert np.nanmax(result.reproj_error) < 40.0
    np.testing.assert_allclose(result.pts3d, pts3d, atol=0.05)
    assert result.meta["source"] == "synthetic"


def test_run_with_calibration(rig, cameras, fly, rng):
    pts3d = fly_motion(rng, n_frames=20)
    pts2d = np.array(cameras.project(pts3d))
    cams0 = perturbed_cameras(rig)

    result = run_from_points2d(
        cams0,
        fly,
        pts2d,
        do_calibrate=True,
        calibrate_kwargs={
            "fixed": ["*.intr", "f.rvec", "f.tvec", "rm.tvec[2]"],
            "bone_prior": False,
            "loss": "linear",
            "max_nfev": 2000,
        },
        max_drops=2,
    )
    # Right-side points (seen by the gauge-anchored right cameras) recover
    # tightly; far-side points are weaker once visibility masking is applied,
    # but the whole pose is still close and reprojects well.
    np.testing.assert_allclose(
        result.pts3d[:, fly.right_idx], pts3d[:, fly.right_idx], atol=1e-2
    )
    np.testing.assert_allclose(result.pts3d, pts3d, atol=0.5)
    assert np.nanmax(result.reproj_error) < 5.0
