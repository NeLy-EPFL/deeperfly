"""Tests for the additive bundle-adjustment extensions: per-observation
confidence weights and a soft bone-length prior. Both must be inert by default
(the existing BA tests already guard that) and behave correctly when enabled.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from helpers import small_rotation

from deeperfly import geometry as geom
from deeperfly.bundle_adjustment import bundle_adjust, core
from deeperfly.cameras import CameraGroup


def make_group(rig, rvecs=None, tvecs=None) -> CameraGroup:
    return CameraGroup.from_arrays(
        rig["names"],
        rig["rvecs"] if rvecs is None else rvecs,
        rig["tvecs"] if tvecs is None else tvecs,
        rig["intrs"],
        rig["dists"],
    )


def perturb(rig, rot_sigma=0.02, trans_sigma=2.0):
    rmats = np.asarray(geom.rvec_to_rmat(rig["rvecs"]))
    rmats0 = np.array([small_rotation(rot_sigma, i) @ r for i, r in enumerate(rmats)])
    rvecs0 = np.asarray(geom.rmat_to_rvec(rmats0))
    tvecs0 = rig["tvecs"] + np.random.default_rng(0).normal(
        scale=trans_sigma, size=rig["tvecs"].shape
    )
    return rvecs0, tvecs0


# -- bone-length Jacobian ----------------------------------------------------


def test_bone_jacobian_matches_finite_difference(rng):
    pi = rng.normal(size=(6, 3))
    pj = rng.normal(size=(6, 3))
    dpi, dpj = (
        np.asarray(a) for a in core._bone_jac_per(jnp.asarray(pi), jnp.asarray(pj))
    )

    def length(a, b):
        return np.asarray(core._bone_len_per(jnp.asarray(a), jnp.asarray(b)))

    eps = 1e-6
    for c in range(3):
        step = np.zeros((6, 3))
        step[:, c] = eps
        fd_i = (length(pi + step, pj) - length(pi - step, pj)) / (2 * eps)
        fd_j = (length(pi, pj + step) - length(pi, pj - step)) / (2 * eps)
        np.testing.assert_allclose(dpi[:, c], fd_i, atol=1e-6)
        np.testing.assert_allclose(dpj[:, c], fd_j, atol=1e-6)


# -- per-observation weights -------------------------------------------------


def test_zero_weight_equivals_dropping_observation(rig):
    """A weight of 0 on an observation must match removing it (NaN)."""
    truth = make_group(rig)
    cloud = np.random.default_rng(1).uniform(-0.5, 0.5, size=(60, 3))
    pts2d = np.array(truth.project(cloud))

    # Corrupt a single observation severely.
    pts2d[2, 7] += np.array([120.0, -90.0])

    rvecs0, tvecs0 = perturb(rig)
    cams0 = make_group(rig, rvecs0, tvecs0)
    fixed = ["*.intr", "f.rvec", "f.tvec"]
    init = cloud.copy()  # identical 3D initialization for both solves

    weights = np.ones(pts2d.shape[:2])
    weights[2, 7] = 0.0
    _, opt_w, _ = bundle_adjust(
        cams0, pts2d, fixed=fixed, pts3d=init, weights=weights, max_nfev=2000
    )

    pts2d_drop = pts2d.copy()
    pts2d_drop[2, 7] = np.nan
    _, opt_drop, _ = bundle_adjust(
        cams0, pts2d_drop, fixed=fixed, pts3d=init, max_nfev=2000
    )

    np.testing.assert_allclose(opt_w.rvecs, opt_drop.rvecs, atol=1e-6)
    np.testing.assert_allclose(opt_w.tvecs, opt_drop.tvecs, atol=1e-6)


def test_downweighting_outlier_improves_recovery(rig):
    truth = make_group(rig)
    cloud = np.random.default_rng(3).uniform(-0.5, 0.5, size=(60, 3))
    pts2d = np.array(truth.project(cloud))
    pts2d[4, 11] += np.array([80.0, 80.0])  # one gross outlier

    rvecs0, tvecs0 = perturb(rig)
    cams0 = make_group(rig, rvecs0, tvecs0)
    fixed = ["*.intr", "f.rvec", "f.tvec"]

    _, opt_plain, _ = bundle_adjust(cams0, pts2d, fixed=fixed, max_nfev=2000)

    weights = np.ones(pts2d.shape[:2])
    weights[4, 11] = 0.0
    _, opt_w, _ = bundle_adjust(
        cams0, pts2d, fixed=fixed, weights=weights, max_nfev=2000
    )

    err_plain = np.linalg.norm(opt_plain.tvecs - rig["tvecs"])
    err_w = np.linalg.norm(opt_w.tvecs - rig["tvecs"])
    assert err_w < err_plain


# -- bone-length prior -------------------------------------------------------


def test_bone_prior_consistent_with_clean_solve(rig):
    """Adding a bone prior at the true lengths leaves a clean solve near-zero."""
    truth = make_group(rig)
    rng = np.random.default_rng(7)
    cloud = rng.uniform(-0.5, 0.5, size=(38, 3))
    pts2d = truth.project(cloud)

    bone_pairs = np.array([[i, i + 1] for i in range(0, 36, 2)])  # arbitrary chain
    bone_targets = np.linalg.norm(
        cloud[bone_pairs[:, 0]] - cloud[bone_pairs[:, 1]], axis=1
    )

    rvecs0, tvecs0 = perturb(rig, rot_sigma=0.01, trans_sigma=1.0)
    cams0 = make_group(rig, rvecs0, tvecs0)
    res, opt, pts3d_opt = bundle_adjust(
        cams0,
        pts2d,
        fixed=["*.intr", "f.rvec", "f.tvec"],
        bone_pairs=bone_pairs,
        bone_targets=bone_targets,
        bone_weight=5.0,
        max_nfev=3000,
    )
    assert res.cost < 1e-6
    got = np.linalg.norm(
        pts3d_opt[bone_pairs[:, 0]] - pts3d_opt[bone_pairs[:, 1]], axis=1
    )
    np.testing.assert_allclose(got, bone_targets, atol=1e-4)


def test_bone_prior_pulls_underconstrained_lengths(rig):
    """With a bad initial scale, the bone prior pulls lengths toward targets."""
    truth = make_group(rig)
    rng = np.random.default_rng(8)
    cloud = rng.uniform(-0.5, 0.5, size=(20, 3))
    pts2d = np.array(truth.project(cloud))
    pts2d[3:] = np.nan  # only 3 cameras observe -> 3D weakly constrained along rays

    bone_pairs = np.array([[i, i + 1] for i in range(0, 18, 2)])
    bone_targets = np.linalg.norm(
        cloud[bone_pairs[:, 0]] - cloud[bone_pairs[:, 1]], axis=1
    )

    # Cameras fixed at truth; only points move. Without bones the depths are
    # underdetermined; the prior should keep bone lengths near target.
    _, _, pts3d_opt = bundle_adjust(
        truth,
        pts2d,
        fixed=["*.rvec", "*.tvec", "*.intr"],
        bone_pairs=bone_pairs,
        bone_targets=bone_targets,
        bone_weight=10.0,
        max_nfev=3000,
    )
    got = np.linalg.norm(
        pts3d_opt[bone_pairs[:, 0]] - pts3d_opt[bone_pairs[:, 1]], axis=1
    )
    # Lengths land much closer to target than a random init would.
    assert np.nanmean(np.abs(got - bone_targets)) < 0.1
