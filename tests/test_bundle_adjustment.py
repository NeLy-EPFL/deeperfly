"""Tests for :mod:`deeperfly.bundle_adjustment`.

Three layers are exercised:

* the packed-state builder (:func:`build_state`) -- index layout, the
  fixed/shared reference grammar and slot compaction;
* the per-observation analytic Jacobian used by the solver, cross-checked
  against central finite differences of the projection model;
* the end-to-end solver, which must drive a perturbed rig back to ~0 reprojection
  cost (a wrong Jacobian or sparsity pattern would stall the TRF step).
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly import geometry as geom
from deeperfly.bundle_adjustment import (
    bundle_adjust,
    bundle_adjust_from_config,
    build_state,
    initialize_pts3d,
)
from deeperfly.bundle_adjustment import core
from deeperfly.cameras import CameraGroup
from helpers import small_rotation


def make_group(rig) -> CameraGroup:
    return CameraGroup.from_arrays(
        rig["names"], rig["rvecs"], rig["tvecs"], rig["intrs"], rig["dists"]
    )


# -- build_state: layout -----------------------------------------------------


def test_build_state_recovers_inputs(rig):
    pts2d = np.zeros((7, 5, 2))
    pts3d = np.arange(15, dtype=float).reshape(5, 3)
    state = build_state(
        rig["rvecs"],
        rig["tvecs"],
        rig["intrs"],
        rig["dists"],
        pts2d,
        rig["names"],
        pts3d=pts3d,
    )
    assert np.allclose(state.values[state.rvecs_idx], rig["rvecs"])
    assert np.allclose(state.values[state.tvecs_idx], rig["tvecs"])
    assert np.allclose(state.values[state.intrs_idx], rig["intrs"])
    assert np.allclose(state.values[state.pts3d_idx], pts3d)
    assert not state.fixed.any()


def test_build_state_default_names_and_triangulation(rig):
    group = make_group(rig)
    cloud = np.random.default_rng(1).uniform(-0.5, 0.5, size=(8, 3))
    pts2d = group.project(cloud)
    state = build_state(rig["rvecs"], rig["tvecs"], rig["intrs"], rig["dists"], pts2d)
    # pts3d omitted -> triangulated from the cameras, recovering the cloud.
    assert np.allclose(state.values[state.pts3d_idx], cloud, atol=1e-6)


# -- build_state: fixed/shared grammar ---------------------------------------


def test_build_state_fixed_mask(rig):
    pts2d = np.zeros((7, 4, 2))
    state = build_state(
        rig["rvecs"],
        rig["tvecs"],
        rig["intrs"],
        rig["dists"],
        pts2d,
        rig["names"],
        fixed=["*.intr", "f.rvec", "rm.tvec[2]"],
    )
    f = rig["names"].index("f")
    rm = rig["names"].index("rm")
    assert state.fixed[state.intrs_idx].all()
    assert state.fixed[state.rvecs_idx[f]].all()
    assert state.fixed[state.tvecs_idx[rm, 2]]
    assert not state.fixed[state.tvecs_idx[rm, 0]]
    assert not state.fixed[state.rvecs_idx[rm]].any()


def test_build_state_shared_compaction(rig):
    pts2d = np.zeros((7, 3, 2))
    shared = [["f.tvec[2]", "lf.tvec[2]", "rf.tvec[2]"]]
    plain = build_state(
        rig["rvecs"], rig["tvecs"], rig["intrs"], rig["dists"], pts2d, rig["names"]
    )
    state = build_state(
        rig["rvecs"],
        rig["tvecs"],
        rig["intrs"],
        rig["dists"],
        pts2d,
        rig["names"],
        shared=shared,
    )
    # three slots collapse to one -> two fewer free parameters
    assert state.values.size == plain.values.size - 2
    f, lf, rf = (rig["names"].index(n) for n in ("f", "lf", "rf"))
    slots = {int(state.tvecs_idx[i, 2]) for i in (f, lf, rf)}
    assert len(slots) == 1


def test_build_state_shared_fixed_interaction(rig):
    pts2d = np.zeros((7, 3, 2))
    state = build_state(
        rig["rvecs"],
        rig["tvecs"],
        rig["intrs"],
        rig["dists"],
        pts2d,
        rig["names"],
        fixed=["f.tvec"],
        shared=[["f.tvec[2]", "lf.tvec[2]"]],
    )
    # fixing f.tvec must propagate to the shared lf.tvec[2] slot
    lf = rig["names"].index("lf")
    assert state.fixed[state.tvecs_idx[lf, 2]]


def test_build_state_shared_warns_on_differing_values(rig):
    rvecs = rig["rvecs"].copy()
    pts2d = np.zeros((7, 3, 2))
    with pytest.warns(UserWarning, match="differing initial values"):
        build_state(
            rvecs,
            rig["tvecs"],
            rig["intrs"],
            rig["dists"],
            pts2d,
            rig["names"],
            shared=[["rh.rvec", "lh.rvec"]],
        )


@pytest.mark.parametrize(
    "ref, match",
    [
        ("frvec", "malformed"),
        ("f.bogus", "unknown parameter"),
        ("nope.rvec", "unknown camera"),
        ("f.rvec[9]", "out of range"),
    ],
)
def test_build_state_invalid_references(rig, ref, match):
    pts2d = np.zeros((7, 2, 2))
    with pytest.raises(ValueError, match=match):
        build_state(
            rig["rvecs"],
            rig["tvecs"],
            rig["intrs"],
            rig["dists"],
            pts2d,
            rig["names"],
            fixed=[ref],
        )


# -- initialize_pts3d --------------------------------------------------------


def test_initialize_pts3d_matches_triangulation(rig):
    group = make_group(rig)
    cloud = np.random.default_rng(2).uniform(-0.5, 0.5, size=(10, 3))
    pts2d = group.project(cloud)
    got = np.asarray(initialize_pts3d(pts2d, rig["rvecs"], rig["tvecs"], rig["intrs"]))
    assert np.allclose(got, cloud, atol=1e-6)


# -- analytic Jacobian vs finite differences ---------------------------------


def test_project_jacobian_matches_finite_difference(rng):
    """The autodiff per-observation Jacobian must match central differences."""
    pt3d = rng.normal(size=3) + np.array([0, 0, 5.0])
    rvec = rng.normal(size=3) * 0.3
    tvec = np.array([0.1, -0.2, 0.3])
    intr = np.array([800.0, 810.0, 320.0, 240.0])
    dist = rng.normal(size=5) * 0.01
    args = [pt3d, rvec, tvec, intr, dist]

    jac = core._jac_per_obs(*[np.asarray(a)[None] for a in args])
    jac = [np.asarray(j)[0] for j in jac]  # each (2, dim)

    def project(a):
        return np.asarray(geom.project_full_one(*a))

    eps = 1e-6
    for k, a in enumerate(args):
        for i in range(a.size):
            step = np.zeros_like(a)
            step[i] = eps
            hi = [*args[:k], a + step, *args[k + 1 :]]
            lo = [*args[:k], a - step, *args[k + 1 :]]
            fd = (project(hi) - project(lo)) / (2 * eps)
            assert np.allclose(jac[k][:, i], fd, atol=1e-5), f"arg {k} elem {i}"


# -- end-to-end solver -------------------------------------------------------


def perturb_extrinsics(rig, rot_sigma=0.05, trans_sigma=5.0):
    rmats = np.asarray(geom.rvec_to_rmat(rig["rvecs"]))
    rmats0 = np.array([small_rotation(rot_sigma, i) @ R for i, R in enumerate(rmats)])
    rvecs0 = np.asarray(geom.rmat_to_rvec(rmats0))
    tvecs0 = rig["tvecs"] + np.random.default_rng(0).normal(
        scale=trans_sigma, size=rig["tvecs"].shape
    )
    return rvecs0, tvecs0


def test_bundle_adjust_recovers_perturbed_rig(rig):
    group = make_group(rig)
    cloud = np.random.default_rng(1).uniform(-0.5, 0.5, size=(120, 3))
    pts2d = group.project(cloud)

    rvecs0, tvecs0 = perturb_extrinsics(rig)
    cams0 = CameraGroup.from_arrays(
        rig["names"], rvecs0, tvecs0, rig["intrs"], rig["dists"]
    )
    res, opt, pts3d_opt = bundle_adjust(cams0, pts2d, fixed=["*.intr"], max_nfev=2000)

    assert res.cost < 1e-6
    assert isinstance(opt, CameraGroup)
    assert pts3d_opt.shape == cloud.shape
    # intrinsics were fixed -> unchanged
    assert np.allclose(opt.intrs, rig["intrs"])
    # reprojection of the refined solution matches the observations
    assert np.allclose(opt.project(pts3d_opt), pts2d, atol=1e-3)


def test_bundle_adjust_missing_observations(rig):
    group = make_group(rig)
    cloud = np.random.default_rng(4).uniform(-0.5, 0.5, size=(80, 3))
    pts2d = np.array(group.project(cloud))
    # drop ~15% of observations
    drop = np.random.default_rng(5).random(pts2d.shape[:2]) < 0.15
    pts2d[drop] = np.nan

    rvecs0, tvecs0 = perturb_extrinsics(rig, rot_sigma=0.02, trans_sigma=2.0)
    cams0 = CameraGroup.from_arrays(
        rig["names"], rvecs0, tvecs0, rig["intrs"], rig["dists"]
    )
    res, _, _ = bundle_adjust(cams0, pts2d, fixed=["*.intr"], max_nfev=2000)
    assert res.cost < 1e-6


def test_bundle_adjust_from_config(rig):
    group = make_group(rig)
    cloud = np.random.default_rng(2).uniform(-0.5, 0.5, size=(120, 3))
    pts2d = group.project(cloud)

    # Perturb all but the anchor `f`; leave tvec[2] (shared+fixed) at ground truth.
    rvecs0, tvecs0 = rig["rvecs"].copy(), rig["tvecs"].copy()
    for i, name in enumerate(rig["names"]):
        if name == "f":
            continue
        rmat = np.asarray(geom.rvec_to_rmat(rig["rvecs"][i]))
        rvecs0[i] = np.asarray(geom.rmat_to_rvec(small_rotation(0.005, i) @ rmat))
        tvecs0[i, :2] += np.random.default_rng(i).normal(scale=0.2, size=2)

    config = {
        "cameras": {
            name: {
                "rvec": rvecs0[i].tolist(),
                "tvec": tvecs0[i].tolist(),
                "focal_length_px": rig["intrs"][i, :2].tolist(),
                "principal_point_px": rig["intrs"][i, 2:].tolist(),
            }
            for i, name in enumerate(rig["names"])
        },
        "bundle_adjustment": {
            "solver": "least_squares_scipy",
            "fixed": ["*.intr", "f.rvec", "f.tvec", "rm.tvec[2]"],
            "shared": [["f.tvec[2]", "lf.tvec[2]", "rf.tvec[2]"]],
            "least_squares_scipy": {"max_nfev": 2000, "loss": "linear"},
        },
    }
    res, opt, _ = bundle_adjust_from_config(config, pts2d)
    assert res.cost < 1e-6
    # anchor camera held exactly fixed
    f = rig["names"].index("f")
    assert np.allclose(opt.rvecs[f], rig["rvecs"][f])
    assert np.allclose(opt.tvecs[f], rig["tvecs"][f])


def test_bundle_adjust_from_config_rejects_unknown_solver(rig):
    config = {
        "cameras": {
            "a": {
                "rvec": [0, 0, 0],
                "tvec": [0, 0, 5],
                "focal_length_px": 800.0,
                "principal_point_px": [1, 2],
            }
        },
        "bundle_adjustment": {"solver": "ceres"},
    }
    with pytest.raises(ValueError, match="unsupported solver"):
        bundle_adjust_from_config(config, np.zeros((1, 1, 2)))
