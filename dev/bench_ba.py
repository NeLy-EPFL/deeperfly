"""Benchmark bundle adjustment via scipy.optimize.least_squares (TRF + LSMR).

Mirrors the setup in examples/bundle_adjustment.ipynb: 7 cameras around the origin, random 3D
points, no distortion, all points visible in all views. The Jacobian for this
problem is highly sparse (each obs touches one camera and one point).

Run as `uv run python dev/bench_ba.py`.
"""

import time

import jax
import numpy as np
from scipy.linalg import expm

from deeperfly import geometry as mvg
from deeperfly.bundle_adjustment import core as ba_jax
from deeperfly.bundle_adjustment.state import build_state

jax.config.update("jax_enable_x64", True)


def get_rmat(yaw):
    y = np.array([0, 0, -1])
    z = np.array([-np.cos(yaw), -np.sin(yaw), 0])
    return np.array([np.cross(y, z), y, z])


def small_rotation(sigma, seed):
    rng = np.random.default_rng(seed)
    omega = rng.normal(scale=sigma, size=3)
    K = np.array(
        [[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]]
    )
    return expm(K)


def setup(n_pts=100, seed=0):
    f, w, h = 22388.125, 1024, 512
    f_mm = 107.463
    cx, cy = (w - 1) / 2, (h - 1) / 2
    azimuths = np.deg2rad([-120, -90, -45, 0, 45, 90, 120])
    intr_gt = np.array([f, cx, cy])
    rmats_gt = np.array([get_rmat(t) for t in azimuths])
    tvecs_gt = np.array([[0, 0, f_mm]] * len(rmats_gt), dtype=float)
    pts3d_gt = np.random.default_rng(seed).uniform(-0.5, 0.5, size=(n_pts, 3))
    rvecs_gt = np.asarray(mvg.rmat_to_rvec(rmats_gt))
    dist_gt = np.array((), dtype=float)
    pts2d_gt = np.asarray(
        mvg.project_full(pts3d_gt, rvecs_gt, tvecs_gt, intr_gt, dist_gt)
    )
    rmats0 = np.array([small_rotation(0.05, i) @ R for i, R in enumerate(rmats_gt)])
    tvecs0 = tvecs_gt + np.random.default_rng(0).normal(scale=5, size=tvecs_gt.shape)
    rvecs0 = np.asarray(mvg.rmat_to_rvec(rmats0))
    return rvecs0, tvecs0, intr_gt, dist_gt, pts2d_gt


def make_args(rvecs0, tvecs0, intr_gt, dist_gt, pts2d_gt):
    """Pack the state with all intrinsics held fixed (extrinsics + points free)."""
    n_views = len(rvecs0)
    intrs = np.broadcast_to(intr_gt, (n_views, intr_gt.size))
    dists = np.broadcast_to(dist_gt, (n_views, dist_gt.size))
    return build_state(rvecs0, tvecs0, intrs, dists, pts2d_gt, fixed=["*.intr"])


def cost_from_residuals(r):
    return 0.5 * float(np.sum(np.asarray(r) ** 2))


def run_scipy(args, max_nfev=1000):
    res, _ = ba_jax.bundle_adjust(*args, max_nfev=max_nfev)
    return res


def time_it(fn, warmup=1, repeat=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), np.median(times)


def main():
    for n_pts in [100, 500, 2000]:
        args = make_args(*setup(n_pts=n_pts))
        n_views = args.rvecs_idx.shape[0]
        n_obs = int(np.isfinite(args.pts2d).all(axis=-1).sum())
        n_free = int((~args.fixed).sum())

        print(
            f"\n=== n_views={n_views}, n_pts={n_pts}, n_obs={n_obs}, "
            f"n_free={n_free}, n_residuals={2 * n_obs} ==="
        )

        res = run_scipy(args)
        print(
            f"  scipy.least_squares: cost={cost_from_residuals(res.fun):.4e}  "
            f"nfev={res.nfev}  status={res.status}"
        )

        tmin, tmed = time_it(lambda: run_scipy(args))
        print(f"  timing: min={tmin * 1e3:8.1f} ms  median={tmed * 1e3:8.1f} ms")


if __name__ == "__main__":
    main()
