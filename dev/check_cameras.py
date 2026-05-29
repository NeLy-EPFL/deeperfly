"""Sanity checks for the config-driven camera / bundle-adjustment API.

Run as `uv run python dev/check_cameras.py`. Verifies, against the reference
`get_rmat` rig used elsewhere in the project:

1. orbit config resolves to the same extrinsics, and projection matches geometry;
2. the fixed/shared parser produces the expected mask and slot compaction;
3. bundle adjustment recovers perturbed cameras (free path and config path).
"""

from pathlib import Path

import numpy as np

from deeperfly import geometry as mvg
from deeperfly.bundle_adjustment import (
    bundle_adjust,
    bundle_adjust_from_config,
    build_state,
)
from deeperfly.cameras import CameraGroup

TOML = Path(__file__).with_name("cameras.toml")
F, F_MM, W, H = 22388.125, 107.463, 1024, 512
AZIMUTHS = [-120, -90, -45, 0, 45, 90, 120]


def get_rmat(yaw):
    y = np.array([0, 0, -1])
    z = np.array([-np.cos(yaw), -np.sin(yaw), 0])
    return np.array([np.cross(y, z), y, z])


def reference_extrinsics():
    rmats = np.array([get_rmat(t) for t in np.deg2rad(AZIMUTHS)])
    rvecs = np.asarray(mvg.rmat_to_rvec(rmats))
    tvecs = np.array([[0.0, 0.0, F_MM]] * len(AZIMUTHS))
    return rvecs, tvecs


def small_rotation(sigma, seed):
    from scipy.linalg import expm

    o = np.random.default_rng(seed).normal(scale=sigma, size=3)
    return expm(np.array([[0, -o[2], o[1]], [o[2], 0, -o[0]], [-o[1], o[0], 0]]))


def check_convention():
    group = CameraGroup.from_config(TOML)
    rvecs_gt, tvecs_gt = reference_extrinsics()

    assert group.names == ["rh", "rm", "rf", "f", "lf", "lm", "lh"]
    assert np.allclose(group.rvecs, rvecs_gt, atol=1e-9), "rvecs differ from get_rmat"
    assert np.allclose(group.tvecs, tvecs_gt, atol=1e-9), "tvecs differ from get_rmat"
    assert np.allclose(group.intrs, [F, F, (W - 1) / 2, (H - 1) / 2])

    pts3d = np.random.default_rng(0).uniform(-0.5, 0.5, size=(50, 3))
    ref = np.asarray(
        mvg.project_full(
            pts3d,
            rvecs_gt,
            tvecs_gt,
            np.array([F, (W - 1) / 2, (H - 1) / 2]),
            np.array(()),
        )
    )
    assert np.allclose(group.project(pts3d), ref, atol=1e-6), "projection mismatch"
    print("[1] convention + projection ......... PASS")


def check_parser():
    group = CameraGroup.from_config(TOML)
    fixed = ["*.intr", "f.rvec", "f.tvec", "rm.tvec[2]"]
    shared = [["f.tvec[2]", "lf.tvec[2]", "rf.tvec[2]"]]

    plain = build_state(
        group.rvecs,
        group.tvecs,
        group.intrs,
        group.dists,
        group.project(np.zeros((3, 3))),
        group.names,
    )
    state = build_state(
        group.rvecs,
        group.tvecs,
        group.intrs,
        group.dists,
        group.project(np.zeros((3, 3))),
        group.names,
        fixed=fixed,
        shared=shared,
    )

    # Three tvec[2] slots collapse to one -> two fewer parameters.
    assert state.values.size == plain.values.size - 2, "sharing did not compact"
    # f/lf/rf all index the same tvec[2] slot now.
    f, lf, rf = (group.names.index(n) for n in ("f", "lf", "rf"))
    s = {
        int(state.tvecs_idx[f, 2]),
        int(state.tvecs_idx[lf, 2]),
        int(state.tvecs_idx[rf, 2]),
    }
    assert len(s) == 1, "f/lf/rf tvec[2] not shared"
    # 7*4 intrinsics + f.rvec(3) + f.tvec(3) all fixed; that shared slot is fixed (f.tvec).
    assert state.fixed[state.intrs_idx].all(), "intrinsics not all fixed"
    assert (
        state.fixed[state.rvecs_idx[f]].all() and state.fixed[state.tvecs_idx[f]].all()
    )
    assert bool(state.fixed[s.pop()]), "shared slot should be fixed via f.tvec"
    print("[2] fixed/shared parsing ............ PASS")


def check_bundle_adjust_free():
    """Perturb every camera; with intrinsics fixed the gauge-free fit -> ~0 cost."""
    group = CameraGroup.from_config(TOML)
    pts3d = np.random.default_rng(1).uniform(-0.5, 0.5, size=(120, 3))
    pts2d = group.project(pts3d)

    rmats0 = np.array([small_rotation(0.05, i) @ c.rmat for i, c in enumerate(group)])
    rvecs0 = np.asarray(mvg.rmat_to_rvec(rmats0))
    tvecs0 = group.tvecs + np.random.default_rng(0).normal(
        scale=5, size=group.tvecs.shape
    )
    cams0 = CameraGroup.from_arrays(
        group.names, rvecs0, tvecs0, group.intrs, group.dists
    )

    res, opt, pts3d_opt = bundle_adjust(cams0, pts2d, fixed=["*.intr"], max_nfev=2000)
    assert res.cost < 1e-6, f"free BA did not converge: cost={res.cost:.3e}"
    assert isinstance(opt, CameraGroup) and pts3d_opt.shape == pts3d.shape
    print(f"[3] bundle_adjust (free) ............ PASS  (cost={res.cost:.2e})")


def check_from_config():
    """Config path: anchor f + share/fix tvec[2]; perturb only the free dofs."""
    group = CameraGroup.from_config(TOML)
    pts3d = np.random.default_rng(2).uniform(-0.5, 0.5, size=(120, 3))
    pts2d = group.project(pts3d)

    # Perturb orientation + xy-translation of every camera except the anchor `f`;
    # leave tvec[2] (shared & fixed in the config) at ground truth so the
    # constrained solution is still exact.
    rvecs0, tvecs0 = group.rvecs.copy(), group.tvecs.copy()
    for i, name in enumerate(group.names):
        if name == "f":
            continue
        rvecs0[i] = mvg.rmat_to_rvec(small_rotation(0.005, i) @ group[name].rmat)
        tvecs0[i, :2] += np.random.default_rng(i).normal(scale=0.2, size=2)
    cams0 = CameraGroup.from_arrays(
        group.names, rvecs0, tvecs0, group.intrs, group.dists
    )

    res, opt, _ = bundle_adjust_from_config(_perturbed_config(cams0), pts2d)
    assert res.cost < 1e-6, (
        f"config BA did not converge: cost={res.cost:.3e} (status={res.status}, nfev={res.nfev})"
    )
    print(f"[4] bundle_adjust_from_config ....... PASS  (cost={res.cost:.2e})")


def _perturbed_config(cams0):
    """Build a config dict whose cameras are `cams0`, reusing the TOML's BA section."""
    import tomllib

    with open(TOML, "rb") as fh:
        cfg = tomllib.load(fh)
    cfg.pop("camera_defaults", None)
    cfg["cameras"] = {
        name: {
            "rvec": cams0[name].rvec.tolist(),
            "tvec": cams0[name].tvec.tolist(),
            "focal_length_px": cams0[name].intr[:2].tolist(),
            "principal_point_px": cams0[name].intr[2:].tolist(),
        }
        for name in cams0.names
    }
    return cfg


if __name__ == "__main__":
    check_convention()
    check_parser()
    check_bundle_adjust_free()
    check_from_config()
    print("\nall checks passed.")
