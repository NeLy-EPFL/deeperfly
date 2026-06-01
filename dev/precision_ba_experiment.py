"""float32 vs float64 bundle-adjustment precision experiment.

Answers: does running BA in float32 (the only thing jax-mps could do, since MLX
is float32-only) degrade the result vs the current float64 path?

Reuses the microscope-like 7-camera rig from dev/bench_ba.py (very long focal,
so projected coords / Jacobians have large dynamic range -- the regime where
single precision is most likely to bite). We run the *same* problem twice:

  * float64 -- current behaviour (jax_enable_x64 is on at import).
  * float32 -- cast the packed state to float32 so the JAX projection + Jacobian
    kernels trace in float32. (scipy's trust-region bookkeeping stays double --
    it forces x0 to float64 -- so this is the *optimistic* float32 case: if it's
    already bad here, full float32 is worse.)

Metrics: final reprojection RMS (px), scipy nfev / status / optimality, and how
far the float32 solution drifts from the float64 one. Two regimes: noise-free
(exposes the raw precision floor) and 1 px detector-like noise (is the float32
penalty below the measurement noise, i.e. does it matter in practice?).
"""

import numpy as np
from scipy.linalg import expm

from deeperfly import geometry as mvg
from deeperfly.bundle_adjustment import core as ba
from deeperfly.bundle_adjustment.state import BAState, build_state


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


def setup(n_pts=100, seed=0, noise_px=0.0):
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
    if noise_px:
        pts2d_gt = pts2d_gt + np.random.default_rng(seed + 1).normal(
            scale=noise_px, size=pts2d_gt.shape
        )
    rmats0 = np.array([small_rotation(0.05, i) @ R for i, R in enumerate(rmats_gt)])
    tvecs0 = tvecs_gt + np.random.default_rng(0).normal(scale=5, size=tvecs_gt.shape)
    rvecs0 = np.asarray(mvg.rmat_to_rvec(rmats0))
    return rvecs0, tvecs0, intr_gt, dist_gt, pts2d_gt


def make_state(rvecs0, tvecs0, intr_gt, dist_gt, pts2d_gt):
    n_views = len(rvecs0)
    intrs = np.broadcast_to(intr_gt, (n_views, intr_gt.size))
    dists = np.broadcast_to(dist_gt, (n_views, dist_gt.size))
    return build_state(rvecs0, tvecs0, intrs, dists, pts2d_gt, fixed=["*.intr"])


def as_f32(state: BAState) -> BAState:
    """Cast the float leaves of a packed state to float32 (indices stay int)."""
    return state._replace(
        values=state.values.astype(np.float32),
        pts2d=state.pts2d.astype(np.float32),
    )


def reproj_rms(result) -> float:
    """RMS reprojection error in pixels from the residual vector."""
    r = np.asarray(result.fun, dtype=np.float64)
    return float(np.sqrt(np.mean(r**2)))


def run(state) -> tuple:
    res, sol = ba.bundle_adjust(*state, max_nfev=1000)
    return res, sol


def sol_vec(sol):
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in sol])


def main():
    for noise_px in (0.0, 1.0):
        tag = "noise-free" if noise_px == 0 else f"{noise_px:g}px detector noise"
        print(f"\n================  {tag}  ================")
        for n_pts in (100, 500):
            st64 = make_state(*setup(n_pts=n_pts, noise_px=noise_px))
            st32 = as_f32(st64)

            res64, sol64 = run(st64)
            res32, sol32 = run(st32)

            rms64, rms32 = reproj_rms(res64), reproj_rms(res32)
            v64, v32 = sol_vec(sol64), sol_vec(sol32)
            drift = float(np.linalg.norm(v32 - v64))
            denom = float(np.linalg.norm(v64)) or 1.0

            print(f"\n  n_pts={n_pts}  (n_resid={res64.fun.size})")
            print(
                f"    float64: rms={rms64:.3e} px  nfev={res64.nfev:>3}  "
                f"status={res64.status}  optimality={res64.optimality:.2e}"
            )
            print(
                f"    float32: rms={rms32:.3e} px  nfev={res32.nfev:>3}  "
                f"status={res32.status}  optimality={res32.optimality:.2e}"
            )
            print(
                f"    rms ratio f32/f64 = {rms32 / max(rms64, 1e-30):.3g}   "
                f"solution drift ||f32-f64|| = {drift:.3e} "
                f"(rel {drift / denom:.2e})"
            )


if __name__ == "__main__":
    main()
