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
from deeperfly.config import Config
from helpers import AZIMUTHS_DEG, CAMERA_NAMES, DISTANCE_MM, small_rotation


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


def test_irls_losses_consistent():
    """The IRLS losses report scipy's documented rho(z), its exact derivative
    as rho'(z), and zero curvature (the point of the IRLS form)."""
    z = np.linspace(0.01, 9, 50)
    expected = {
        "huber": np.where(z <= 1, z, 2 * z**0.5 - 1),
        "cauchy": np.log1p(z),
        "arctan": np.arctan(z),
    }
    for name, loss in core._IRLS_LOSSES.items():
        rho = loss(z)
        assert np.allclose(rho[0], expected[name]), name
        fd = (loss(z + 1e-7)[0] - loss(z - 1e-7)[0]) / 2e-7
        assert np.allclose(rho[1], fd, atol=1e-6), name
        assert (rho[1] > 0).all() and (rho[2] == 0).all(), name


def test_bundle_adjust_huber_outliers(rig):
    """Gross outliers drag a linear fit; the same fit with huber loss recovers.

    Exercises the IRLS translation of ``loss="huber"``: scipy's native huber
    rescaling zeroes the Jacobian rows of outlier residuals, which stalls the
    sparse lsmr solver even from a good initialization.
    """
    group = make_group(rig)
    cloud = np.random.default_rng(6).uniform(-0.5, 0.5, size=(60, 3))
    pts2d = np.asarray(group.project(cloud))
    out = np.random.default_rng(7)
    out_view, out_pt = np.unravel_index(
        out.choice(pts2d.size // 2, size=40, replace=False), pts2d.shape[:2]
    )
    angle = out.uniform(0, 2 * np.pi, size=40)
    magnitude = out.uniform(30, 150, size=40)
    corrupt = pts2d.copy()
    corrupt[out_view, out_pt] += magnitude[:, None] * np.stack(
        [np.cos(angle), np.sin(angle)], axis=-1
    )

    rvecs0, tvecs0 = perturb_extrinsics(rig, rot_sigma=0.02, trans_sigma=2.0)
    cams0 = CameraGroup.from_arrays(
        rig["names"], rvecs0, tvecs0, rig["intrs"], rig["dists"]
    )
    inlier = np.ones(pts2d.shape[:2], dtype=bool)
    inlier[out_view, out_pt] = False

    def median_inlier_error(cams, pts3d):
        resid = np.asarray(cams.project(pts3d)) - pts2d
        return np.median(np.linalg.norm(resid[inlier], axis=-1))

    # the linear fit is dragged off by the outliers
    res_lin, cams_lin, pts3d_lin = bundle_adjust(
        cams0, corrupt, fixed=["*.intr"], max_nfev=2000
    )
    assert median_inlier_error(cams_lin, pts3d_lin) > 3.0

    # the huber fit pulls the inlier reprojections back to ~0
    res, opt, pts3d_opt = bundle_adjust(
        cams0, corrupt, fixed=["*.intr"], max_nfev=2000, loss="huber", f_scale=1.0
    )
    assert res.status > 0  # converged, not out of budget
    assert median_inlier_error(opt, pts3d_opt) < 0.5


def test_bundle_adjust_from_config(rig):
    group = make_group(rig)
    cloud = np.random.default_rng(2).uniform(-0.5, 0.5, size=(120, 3))
    pts2d = group.project(cloud)

    # Perturb the orbit angles of all but the anchor `f`; keep `distance` at
    # ground truth so tvec[2] (shared+fixed) starts there too (tvec == [0, 0, d]
    # for any orbit camera looking at the origin).
    perturb = np.random.default_rng(3)
    angles0 = {
        name: {"azimuth_deg": az}
        if name == "f"
        else {
            "azimuth_deg": az + perturb.normal(scale=0.5),
            "elevation_deg": perturb.normal(scale=0.5),
            "roll_deg": perturb.normal(scale=0.5),
        }
        for name, az in zip(CAMERA_NAMES, AZIMUTHS_DEG)
    }
    config = {
        "cameras": {
            name: {
                **angles0[name],
                "distance": DISTANCE_MM,
                "focal_length_px": rig["intrs"][i, :2].tolist(),
                "principal_point_px": rig["intrs"][i, 2:].tolist(),
            }
            for i, name in enumerate(rig["names"])
        },
        "pipeline": {
            "bundle_adjustment": {
                "fixed": ["*.intr", "f.rvec", "f.tvec", "rm.tvec[2]"],
                "shared": [["f.tvec[2]", "lf.tvec[2]", "rf.tvec[2]"]],
                # scipy least_squares kwargs sit flat in the table (no solver sub-table).
                "max_nfev": 2000,
                "loss": "linear",
            },
        },
    }
    res, opt, _ = bundle_adjust_from_config(Config.from_dict(config), pts2d)
    assert res.cost < 1e-6
    # anchor camera held exactly fixed
    f = rig["names"].index("f")
    assert np.allclose(opt.rvecs[f], rig["rvecs"][f])
    assert np.allclose(opt.tvecs[f], rig["tvecs"][f])
