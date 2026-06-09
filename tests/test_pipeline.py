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
from deeperfly.pipeline import (
    _resolve_triangulation,
    calibrate,
    reconstruct,
    reconstruct_ransac,
    run_from_points2d,
)
from deeperfly.skeleton import Skeleton
from helpers import leg_indices, small_rotation


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


def test_reconstruct_ransac_rejects_outliers_and_handles_nan(cameras, rng):
    pts3d = fly_motion(rng)
    pts2d = np.array(cameras.project(pts3d))  # (V, T, N, 2)
    pts2d[1, 0, 5] += [200.0, -150.0]  # gross single-view outliers
    pts2d[4, 3, 20] += [-300.0, 120.0]
    pts2d[0, :, 7] = np.nan  # camera 0 never sees point 7 (NaN observations)

    recovered, cleaned, err = reconstruct_ransac(cameras, pts2d, threshold=15.0)
    assert recovered.shape == pts3d.shape
    assert not np.isnan(recovered).any()  # every point still has >= 2 good views
    assert np.nanmax(err) < 15.0  # outliers excluded from the cleaned reprojection
    np.testing.assert_allclose(recovered, pts3d, atol=1e-6)
    assert np.isnan(cleaned[1, 0, 5]).all()  # the outlier observation was rejected
    assert np.isnan(cleaned[0, :, 7]).all()  # the unobserved view stays NaN


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


def test_front_camera_bridges_left_right_in_calibration(rig, cameras, fly, rng):
    """The front camera, seeing both body sides, is what co-registers the two
    camera clusters in bundle adjustment.

    With realistic per-side visibility the right cameras observe only right
    joints and the left cameras only left joints -- disjoint sets. The relative
    pose between the two clusters is then unobservable *unless* some camera sees
    both sides. The front camera does (it runs as two passes), so a wrong
    left-vs-right relative pose is correctable only when its cross-side
    observations are present.
    """
    from deeperfly import geometry as geom
    from deeperfly.triangulate import apply_visibility

    names = rig["names"]
    pts3d = fly_motion(rng, n_frames=16)
    pts2d_full = np.array(cameras.project(pts3d))  # every camera sees everything
    conf = np.ones(pts2d_full.shape[:3])

    # Perturb the three left cameras: rotate the whole cluster ~8 deg about the
    # world +z axis -- a wrong left-vs-right relative pose for BA to recover.
    a = np.deg2rad(8.0)
    Rd = np.array(
        [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0, 0, 1]]
    )
    rvecs, tvecs = rig["rvecs"].copy(), rig["tvecs"].copy()
    for nm in ("lf", "lm", "lh"):
        i = names.index(nm)
        R = np.asarray(geom.rvec_to_rmat(rig["rvecs"][i]))
        center = -R.T @ rig["tvecs"][i]
        Rp = R @ Rd.T
        rvecs[i] = np.asarray(geom.rmat_to_rvec(Rp))
        tvecs[i] = -Rp @ (Rd @ center)
    perturbed = CameraGroup.from_arrays(names, rvecs, tvecs, rig["intrs"], rig["dists"])

    # Free only the left cameras; anchor everyone else and all intrinsics.
    fixed = ["*.intr", "*.dist"]
    for nm in ("rh", "rm", "rf", "f"):
        fixed += [f"{nm}.rvec", f"{nm}.tvec"]

    def left_orientation_error(group) -> float:
        errs = []
        for nm in ("lf", "lm", "lh"):
            rt = np.asarray(geom.rvec_to_rmat(cameras[nm].rvec))
            rr = np.asarray(geom.rvec_to_rmat(group[nm].rvec))
            cos = (np.trace(rt @ rr.T) - 1) / 2
            errs.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        return float(np.mean(errs))

    def run(front_sees_both: bool) -> float:
        pts2d = apply_visibility(pts2d_full.copy(), fly, names)
        if not front_sees_both:  # drop the front camera's left-side observations
            fi = names.index("f")
            for j in leg_indices(fly, "l"):
                pts2d[fi, :, j] = np.nan
        opt, _ = calibrate(
            perturbed,
            pts2d,
            conf,
            fly,
            fixed=fixed,
            bone_prior=False,
            max_frames=16,
            max_nfev=300,
        )
        return left_orientation_error(opt)

    err_both = run(front_sees_both=True)
    err_right_only = run(front_sees_both=False)
    # The bridge lets BA pull the left cluster back to truth; without it the two
    # sides stay misaligned (the ~8 deg error is left largely uncorrected).
    assert err_both < 0.5
    assert err_right_only > 1.0


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
        fps=100.0,
        meta={"source": "synthetic"},
    )
    assert isinstance(result, PoseResult)
    assert result.pts3d.shape == pts3d.shape
    assert not np.isnan(result.pts3d).any()  # every fly point recoverable
    assert np.nanmax(result.reproj_error) < 40.0
    np.testing.assert_allclose(result.pts3d, pts3d, atol=0.05)
    assert result.meta["source"] == "synthetic"


def test_calibrate_legs_only_ignores_corrupted_nonleg(rig, cameras, fly, rng):
    """Restricting BA to the leg joints refines the cameras from those alone, so
    gross errors on the antennae / stripes do not corrupt the calibration."""
    legs = np.concatenate(  # the 30 leg-joint indices
        [leg_indices(fly, "r"), leg_indices(fly, "l")]
    )
    pts3d = fly_motion(rng, n_frames=20)
    pts2d = np.array(cameras.project(pts3d))
    nonleg = np.setdiff1d(np.arange(38), legs)
    pts2d[:, :, nonleg] += rng.normal(scale=200.0, size=pts2d[:, :, nonleg].shape)
    cams0 = perturbed_cameras(rig)

    opt, _ = calibrate(
        cams0,
        pts2d,
        skeleton=fly,
        ba_keypoints=legs,
        fixed=["*.intr", "f.rvec", "f.tvec", "rm.tvec[2]"],
        bone_prior=False,
        loss="linear",
        max_nfev=2000,
    )
    # Despite the corrupted non-leg observations, the leg joints reproject almost
    # exactly -- the cameras were recovered from the legs alone.
    proj = np.asarray(opt.project(pts3d))
    assert np.nanmax(np.abs(proj[:, :, legs] - pts2d[:, :, legs])) < 1e-2


def test_run_with_calibration(rig, cameras, fly):
    # Unlike calibrate() on full observations, run_from_points2d applies
    # per-side visibility masking *before* calibrating, and with bone_prior=False
    # the far side is bridged only by the front camera -- a weakly constrained
    # sub-problem whose conditioning depends on the geometry of the (random)
    # points each camera happens to see. default_rng(0) is degenerate for this
    # rig (a far-side point loses all but one view), so pin a well-conditioned
    # seed; the recovered physics is the same for any non-degenerate cloud.
    rng = np.random.default_rng(5)
    pts3d = fly_motion(rng, n_frames=20)
    pts2d = np.array(cameras.project(pts3d))
    cams0 = perturbed_cameras(rig)

    result = run_from_points2d(
        cams0,
        fly,
        pts2d,
        do_calibrate=True,
        calibrate_kwargs={
            # ba_keypoints defaults to None -> calibrate on all points
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
    right = leg_indices(fly, "r")
    np.testing.assert_allclose(result.pts3d[:, right], pts3d[:, right], atol=1e-2)
    np.testing.assert_allclose(result.pts3d, pts3d, atol=0.5)
    assert np.nanmax(result.reproj_error) < 5.0


# -- triangulation choices (ransac default, greedy, dlt) + pictorial flag -----


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("ransac", "ransac"),
        ("greedy", "greedy"),
        ("dlt", "dlt"),
        ("reproject", "greedy"),  # legacy alias
        ("none", "dlt"),  # legacy alias
    ],
)
def test_resolve_triangulation(spec, expected):
    assert _resolve_triangulation(spec) == expected


@pytest.mark.parametrize("spec", ["bogus", "pictorial", "ransac+greedy", ""])
def test_resolve_triangulation_rejects_bad(spec):
    with pytest.raises(ValueError, match="triangulation"):
        _resolve_triangulation(spec)


@pytest.mark.parametrize("triangulation", ["ransac", "greedy", "dlt"])
def test_run_triangulation_choices(cameras, fly, rng, triangulation):
    pts3d = fly_motion(rng, n_frames=8)
    pts2d = np.array(cameras.project(pts3d))
    if triangulation != "dlt":  # dlt has no outlier handling, so keep it clean
        pts2d[2, 4, 9] += [200.0, 200.0]  # a gross single-view outlier

    result = run_from_points2d(
        cameras,
        fly,
        pts2d,
        do_calibrate=False,
        triangulation=triangulation,
    )
    assert result.meta["triangulation"] == triangulation
    assert result.meta["pictorial"] is False
    assert not np.isnan(result.pts3d).any()
    np.testing.assert_allclose(result.pts3d, pts3d, atol=1e-4)


def test_run_default_triangulation_is_ransac(cameras, fly, rng):
    pts2d = np.array(cameras.project(fly_motion(rng, n_frames=4)))
    result = run_from_points2d(cameras, fly, pts2d, do_calibrate=False)
    assert result.meta["triangulation"] == "ransac"  # the default


def test_run_unknown_triangulation_raises(cameras, fly, rng):
    pts2d = np.array(cameras.project(fly_motion(rng, n_frames=2)))
    with pytest.raises(ValueError, match="triangulation"):
        run_from_points2d(
            cameras, fly, pts2d, do_calibrate=False, triangulation="bogus"
        )


@pytest.mark.parametrize("triangulation", ["ransac", "greedy", "dlt"])
def test_run_pictorial_then_triangulator(cameras, fly, rng, triangulation):
    """pictorial recovers the peak, then the chosen triangulation fits the 3D."""
    from deeperfly import pictorial

    pts3d = fly_motion(rng, n_frames=5)
    proj = np.asarray(cameras.project(pts3d))  # (V, T, 38, 2)
    v, t = proj.shape[:2]
    xy = np.full((v, t, 38, 2, 2), np.nan)
    sc = np.zeros((v, t, 38, 2))
    xy[:, :, :, 0] = proj  # the true projection is the (only) candidate peak
    sc[:, :, :, 0] = 0.9
    cands = pictorial.Candidates(xy=xy, score=sc)

    result = run_from_points2d(
        cameras,
        fly,
        proj,
        np.ones(proj.shape[:3]),
        candidates=cands,
        do_pictorial=True,
        triangulation=triangulation,
        do_calibrate=False,
    )
    assert result.meta["pictorial"] is True
    assert result.meta["triangulation"] == triangulation
    assert result.skeleton.n_points == 38
    np.testing.assert_allclose(result.pts3d[:, 16:19], pts3d[:, 16:19], atol=1e-2)


def test_run_pictorial_requires_candidates(cameras, fly, rng):
    pts2d = np.array(cameras.project(fly_motion(rng, n_frames=2)))
    with pytest.raises(ValueError, match="requires candidates"):
        run_from_points2d(cameras, fly, pts2d, do_calibrate=False, do_pictorial=True)
