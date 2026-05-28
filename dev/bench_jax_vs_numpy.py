"""Benchmark numpy vs jax implementations of geometry primitives
for V=7 views and N=200 points. Run as `uv run python dev/bench_jax_vs_numpy.py`.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from deeperfly.np.geometry import (
    distort,
    intr_to_kmat,
    project,
    rmat_to_rvec,
    rvec_to_rmat,
    triangulate_dlt,
)

jax.config.update("jax_enable_x64", True)

V, N = 7, 200
N_REPEAT = 200
N_WARMUP = 5


# -- JAX ports ---------------------------------------------------------------


def _project_jax(pts3d, pmats):
    pts2dh = pmats[:, :, :3] @ pts3d.T + pmats[:, :, 3:]
    return (pts2dh[:, :2] / pts2dh[:, 2:]).transpose(0, 2, 1)


def _rvec_to_rmat_jax(rvec):
    theta2 = jnp.einsum("...i,...i->...", rvec, rvec)
    theta = jnp.sqrt(theta2)
    small = theta2 < 1e-8
    a = jnp.where(
        small,
        1 - theta2 / 6 + theta2**2 / 120,
        jnp.sin(theta) / jnp.where(small, 1.0, theta),
    )
    b = jnp.where(
        small,
        0.5 - theta2 / 24 + theta2**2 / 720,
        (1 - jnp.cos(theta)) / jnp.where(small, 1.0, theta2),
    )
    zero = jnp.zeros_like(rvec[..., 0])
    W = jnp.stack(
        [
            jnp.stack([zero, -rvec[..., 2], rvec[..., 1]], axis=-1),
            jnp.stack([rvec[..., 2], zero, -rvec[..., 0]], axis=-1),
            jnp.stack([-rvec[..., 1], rvec[..., 0], zero], axis=-1),
        ],
        axis=-2,
    )
    vvt = rvec[..., :, None] * rvec[..., None, :]
    diag = (1 - b * theta2)[..., None, None] * jnp.eye(3)
    return diag + a[..., None, None] * W + b[..., None, None] * vvt


def _rmat_to_rvec_jax(rmat):
    r00, r01, r02 = rmat[..., 0, 0], rmat[..., 0, 1], rmat[..., 0, 2]
    r10, r11, r12 = rmat[..., 1, 0], rmat[..., 1, 1], rmat[..., 1, 2]
    r20, r21, r22 = rmat[..., 2, 0], rmat[..., 2, 1], rmat[..., 2, 2]
    rho = jnp.stack((r21 - r12, r02 - r20, r10 - r01), axis=-1)
    s = jnp.linalg.norm(rho, axis=-1) / 2
    c = jnp.clip((r00 + r11 + r22 - 1) / 2, -1.0, 1.0)
    theta = jnp.arccos(c)
    near_pi = s < 1e-5
    rvec = rho * (theta / jnp.where(near_pi, 1.0, 2.0 * s))[..., None]
    ax = jnp.sqrt(jnp.maximum((r00 + 1.0) / 2, 0.0))
    ay = jnp.sqrt(jnp.maximum((r11 + 1.0) / 2, 0.0)) * jnp.where(r01 < 0, -1.0, 1.0)
    az = jnp.sqrt(jnp.maximum((r22 + 1.0) / 2, 0.0)) * jnp.where(r02 < 0, -1.0, 1.0)
    flip = (
        (jnp.abs(ax) < jnp.abs(ay))
        & (jnp.abs(ax) < jnp.abs(az))
        & ((r12 > 0) != (ay * az > 0))
    )
    az = jnp.where(flip, -az, az)
    axis = jnp.stack((ax, ay, az), axis=-1)
    norm = jnp.linalg.norm(axis, axis=-1)
    rvec_pi = axis * (theta / jnp.where(norm == 0, 1.0, norm))[..., None]
    rvec_deg = jnp.where((c > 0)[..., None], 0.0, rvec_pi)
    return jnp.where(near_pi[..., None], rvec_deg, rvec)


def _distort_jax(pts2d, dists):
    # Mirror numpy version for n_params == 5 (k1, k2, p1, p2, k3).
    x = pts2d[..., 0]
    y = pts2d[..., 1]
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    k1, k2, p1, p2, k3 = (
        dists[..., 0],
        dists[..., 1],
        dists[..., 2],
        dists[..., 3],
        dists[..., 4],
    )
    # Broadcast dists (V,) over points (V, N).
    mult = 1.0 + k1[:, None] * r2 + k2[:, None] * r4 + k3[:, None] * r6
    xy = x * y
    add_x = 2 * p1[:, None] * xy + p2[:, None] * (r2 + 2 * x * x)
    add_y = p1[:, None] * (r2 + 2 * y * y) + 2 * p2[:, None] * xy
    return jnp.stack([x * mult + add_x, y * mult + add_y], axis=-1)


def _triangulate_dlt_jax(pts2d, pmats):
    a = jnp.einsum("vni,vj->nvij", pts2d, pmats[:, 2]) - pmats[:, :2]
    valid = jnp.moveaxis(jnp.isfinite(pts2d).all(axis=-1), 0, -1)
    a = jnp.where(valid[..., None, None], a, 0.0)
    a = a.reshape((a.shape[0], -1, 4))
    ata = jnp.einsum("...ij,...ik->...jk", a, a)
    pts3dh = jnp.linalg.eigh(ata)[1][..., :, 0]
    result = pts3dh[..., :3] / pts3dh[..., 3:]
    return jnp.where((valid.sum(axis=-1) < 2)[..., None], jnp.nan, result)


project_jit = jax.jit(_project_jax)
rvec_to_rmat_jit = jax.jit(_rvec_to_rmat_jax)
rmat_to_rvec_jit = jax.jit(_rmat_to_rvec_jax)
distort_jit = jax.jit(_distort_jax)
triangulate_dlt_jit = jax.jit(_triangulate_dlt_jax)


# -- Inputs ------------------------------------------------------------------

rng = np.random.default_rng(0)
rvecs = rng.normal(size=(V, 3)) * 0.3
rmats = rvec_to_rmat(rvecs)
tvecs = rng.normal(size=(V, 3))
intrs = np.tile(np.array([800.0, 800.0, 320.0, 240.0]), (V, 1))
kmats = intr_to_kmat(intrs)
rtmats = np.concatenate((rmats, tvecs[..., None]), axis=-1)
pmats = kmats @ rtmats  # (V, 3, 4)
pts3d = rng.normal(size=(N, 3)) + np.array([0.0, 0.0, 5.0])
pts2d = project(pts3d, pmats)  # (V, N, 2)
dists = rng.normal(size=(V, 5)) * 0.01

pmats_j, pts3d_j = jnp.asarray(pmats), jnp.asarray(pts3d)
pts2d_j, dists_j = jnp.asarray(pts2d), jnp.asarray(dists)
rvecs_j, rmats_j = jnp.asarray(rvecs), jnp.asarray(rmats)


# -- Bench helpers -----------------------------------------------------------


def bench_np(fn, *args):
    for _ in range(N_WARMUP):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        fn(*args)
    return (time.perf_counter() - t0) / N_REPEAT


def bench_jax(fn, *args):
    for _ in range(N_WARMUP):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) / N_REPEAT


def bench_jax_with_transfer(fn, *args):
    """Includes np→jax transfer cost (realistic when caller holds numpy arrays)."""
    np_args = [np.asarray(a) for a in args]
    for _ in range(N_WARMUP):
        jax.block_until_ready(fn(*(jnp.asarray(a) for a in np_args)))
    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        jax.block_until_ready(fn(*(jnp.asarray(a) for a in np_args)))
    return (time.perf_counter() - t0) / N_REPEAT


cases = [
    (
        "project",
        lambda: project(pts3d, pmats),
        lambda: project_jit(pts3d_j, pmats_j),
        (pts3d, pmats),
    ),
    (
        "triangulate_dlt",
        lambda: triangulate_dlt(pts2d, pmats),
        lambda: triangulate_dlt_jit(pts2d_j, pmats_j),
        (pts2d, pmats),
    ),
    (
        "rvec_to_rmat",
        lambda: rvec_to_rmat(rvecs),
        lambda: rvec_to_rmat_jit(rvecs_j),
        (rvecs,),
    ),
    (
        "rmat_to_rvec",
        lambda: rmat_to_rvec(rmats),
        lambda: rmat_to_rvec_jit(rmats_j),
        (rmats,),
    ),
    (
        "distort",
        lambda: distort(pts2d, dists),
        lambda: distort_jit(pts2d_j, dists_j),
        (pts2d, dists),
    ),
]

print(
    f"{'function':<18} {'numpy (µs)':>12} {'jax-pre (µs)':>14} {'jax+xfer (µs)':>16} {'speedup-pre':>12} {'speedup-xfer':>13}"
)
print("-" * 92)

for name, np_fn, jax_fn, np_args in cases:
    t_np = bench_np(np_fn)
    t_jx = bench_jax(jax_fn)
    # Build transfer-included variant per case
    if name == "project":
        t_xf = bench_jax_with_transfer(project_jit, pts3d, pmats)
    elif name == "triangulate_dlt":
        t_xf = bench_jax_with_transfer(triangulate_dlt_jit, pts2d, pmats)
    elif name == "rvec_to_rmat":
        t_xf = bench_jax_with_transfer(rvec_to_rmat_jit, rvecs)
    elif name == "rmat_to_rvec":
        t_xf = bench_jax_with_transfer(rmat_to_rvec_jit, rmats)
    elif name == "distort":
        t_xf = bench_jax_with_transfer(distort_jit, pts2d, dists)
    print(
        f"{name:<18} {t_np * 1e6:>12.2f} {t_jx * 1e6:>14.2f} {t_xf * 1e6:>16.2f} "
        f"{t_np / t_jx:>11.2f}x {t_np / t_xf:>12.2f}x"
    )

# -- Correctness sanity ------------------------------------------------------
print()
print("Max abs diff (numpy vs jax):")
print(
    f"  project:         {np.max(np.abs(project(pts3d, pmats) - np.asarray(project_jit(pts3d_j, pmats_j)))):.2e}"
)
print(
    f"  triangulate_dlt: {np.nanmax(np.abs(triangulate_dlt(pts2d, pmats) - np.asarray(triangulate_dlt_jit(pts2d_j, pmats_j)))):.2e}"
)
print(
    f"  rvec_to_rmat:    {np.max(np.abs(rvec_to_rmat(rvecs) - np.asarray(rvec_to_rmat_jit(rvecs_j)))):.2e}"
)
print(
    f"  rmat_to_rvec:    {np.max(np.abs(rmat_to_rvec(rmats) - np.asarray(rmat_to_rvec_jit(rmats_j)))):.2e}"
)
print(
    f"  distort:         {np.max(np.abs(distort(pts2d, dists) - np.asarray(distort_jit(pts2d_j, dists_j)))):.2e}"
)
