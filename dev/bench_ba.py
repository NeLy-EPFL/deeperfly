"""Benchmark bundle adjustment: scipy.optimize.least_squares vs optimistix LM+LSMR.

Mirrors the setup in dev/test_ba.ipynb: 7 cameras around the origin, 100 random
3D points, no distortion, all points visible in all views. The Jacobian for this
problem is highly sparse (each obs touches one camera and one point).

Run as `uv run python dev/bench_ba.py`.
"""

import time

import jax
import numpy as np
from scipy.linalg import expm

from deeperfly.bundle_adjustment import init
from deeperfly.bundle_adjustment import core as ba_jax
from deeperfly import geometry as mvg

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


def cost_from_residuals(r):
    return 0.5 * float(np.sum(np.asarray(r) ** 2))


def run_scipy(args, max_nfev=1000):
    res, _ = ba_jax.bundle_adjust_scipy(*args, max_nfev=max_nfev)
    return res


def run_optx(
    args, max_steps=1000, linear_solver="lsmr", lsmr_rtol=1e-6, lsmr_atol=1e-6
):
    sol, _ = ba_jax.bundle_adjust_optx(
        *args,
        max_steps=max_steps,
        linear_solver=linear_solver,
        lsmr_rtol=lsmr_rtol,
        lsmr_atol=lsmr_atol,
    )
    jax.block_until_ready(sol.value)
    return sol


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
        rvecs0, tvecs0, intr_gt, dist_gt, pts2d_gt = setup(n_pts=n_pts)
        args = init.prep_args(pts2d_gt, rvecs0, tvecs0, intr_gt, dist_gt)
        n_views = len(rvecs0)
        n_obs = int(np.isfinite(pts2d_gt).all(axis=-1).sum())
        n_free = int((~args[1]).sum())

        print(
            f"\n=== n_views={n_views}, n_pts={n_pts}, n_obs={n_obs}, "
            f"n_free={n_free}, n_residuals={2 * n_obs} ==="
        )

        # Single solve to inspect convergence / final cost
        res_scipy = run_scipy(args)
        cost_scipy = cost_from_residuals(res_scipy.fun)

        sol_optx = run_optx(args)
        # Re-evaluate residuals at sol.value via the existing scipy path
        # by calling the same residuals function inside bundle_adjust_optx.
        # Easier: call run_optx and inspect sol.stats; for final cost we
        # rebuild and project once.
        values, fixed, rv_i, tv_i, in_i, di_i, p3_i, pts2d = args
        full_opt = values.copy()
        full_opt[~fixed] = np.asarray(sol_optx.value)
        rvecs_opt = full_opt[rv_i]
        tvecs_opt = full_opt[tv_i]
        intrs_opt = full_opt[in_i]
        dists_opt = full_opt[di_i]
        pts3d_opt = full_opt[p3_i]
        obs_v, obs_n = np.where(np.isfinite(pts2d).all(axis=-1))
        proj = np.asarray(
            jax.vmap(ba_jax.project_full_one)(
                pts3d_opt[obs_n],
                rvecs_opt[obs_v],
                tvecs_opt[obs_v],
                intrs_opt[obs_v],
                dists_opt[obs_v],
            )
        )
        resid_optx = (proj - pts2d[obs_v, obs_n]).ravel()
        cost_optx = cost_from_residuals(resid_optx)

        print(
            f"  scipy.least_squares: cost={cost_scipy:.4e}  nfev={res_scipy.nfev}  "
            f"status={res_scipy.status}"
        )
        print(
            f"  optimistix LM+LSMR:  cost={cost_optx:.4e}  "
            f"steps={int(sol_optx.stats.get('num_steps', -1))}  "
            f"result={sol_optx.result}"
        )

        # Timing (with warmup so JIT compile is excluded)
        t_scipy_min, t_scipy_med = time_it(lambda: run_scipy(args))
        t_lsmr6_min, t_lsmr6_med = time_it(lambda: run_optx(args))
        t_lsmr3_min, t_lsmr3_med = time_it(
            lambda: run_optx(args, lsmr_rtol=1e-3, lsmr_atol=1e-3)
        )
        t_ncg_min, t_ncg_med = time_it(
            lambda: run_optx(
                args, linear_solver="normal_cg", lsmr_rtol=1e-3, lsmr_atol=1e-3
            )
        )
        rows = [
            ("scipy.lsmr", t_scipy_min, t_scipy_med),
            ("optx LSMR  tol=1e-6", t_lsmr6_min, t_lsmr6_med),
            ("optx LSMR  tol=1e-3", t_lsmr3_min, t_lsmr3_med),
            ("optx N-CG  tol=1e-3", t_ncg_min, t_ncg_med),
        ]
        for label, tmin, tmed in rows:
            print(
                f"  {label:<22} min={tmin * 1e3:8.1f} ms  median={tmed * 1e3:8.1f} ms  "
                f"({t_scipy_med / tmed:.2f}x scipy)"
            )


if __name__ == "__main__":
    main()
