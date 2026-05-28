"""Benchmark manual batched JAX impls vs vmap of *_one siblings.

Compares ``rvec_to_rmat``, ``rmat_to_rvec``, ``distort``, ``project_full``
in ``deeperfly.jax.geometry`` against equivalent ``vmap``-of-``*_one``
versions, under ``jax.jit``, at the typical BA shape (V=7, N=2000).

The library functions are now themselves built via ``vmap``, so the
"manual" column here re-creates the previous hand-batched impls inline
to keep the historical comparison meaningful.

Run as ``uv run python dev/bench_manual_vs_vmap.py``.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from deeperfly.geometry import (
    _NEAR_PI_SIN_THRESH,
    _SMALL_THETA_SQ,
    distort,
    project_full,
    rmat_to_rvec,
    rvec_to_rmat,
)

jax.config.update("jax_enable_x64", True)

V, N = 7, 2000
N_REPEAT = 500
N_WARMUP = 10


# -- old manually-batched impls (inlined for comparison) ---------------------


def rvec_to_rmat_manual(rvec):
    theta_sq = jnp.einsum("...i,...i->...", rvec, rvec)
    theta = jnp.sqrt(theta_sq)
    small = theta_sq < _SMALL_THETA_SQ
    sinc_theta = jnp.where(
        small,
        1 - theta_sq / 6 + theta_sq**2 / 120,
        jnp.sin(theta) / jnp.where(small, 1.0, theta),
    )
    cosc_theta = jnp.where(
        small,
        0.5 - theta_sq / 24 + theta_sq**2 / 720,
        (1 - jnp.cos(theta)) / jnp.where(small, 1.0, theta_sq),
    )
    zero = jnp.zeros_like(rvec[..., 0])
    skew = jnp.stack(
        [
            jnp.stack([zero, -rvec[..., 2], rvec[..., 1]], axis=-1),
            jnp.stack([rvec[..., 2], zero, -rvec[..., 0]], axis=-1),
            jnp.stack([-rvec[..., 1], rvec[..., 0], zero], axis=-1),
        ],
        axis=-2,
    )
    outer = rvec[..., :, None] * rvec[..., None, :]
    diag = (1 - cosc_theta * theta_sq)[..., None, None] * jnp.eye(3)
    return (
        diag + sinc_theta[..., None, None] * skew + cosc_theta[..., None, None] * outer
    )


def rmat_to_rvec_manual(rmat):
    r00, r01, r02 = rmat[..., 0, 0], rmat[..., 0, 1], rmat[..., 0, 2]
    r10, r11, r12 = rmat[..., 1, 0], rmat[..., 1, 1], rmat[..., 1, 2]
    r20, r21, r22 = rmat[..., 2, 0], rmat[..., 2, 1], rmat[..., 2, 2]
    rho = jnp.stack((r21 - r12, r02 - r20, r10 - r01), axis=-1)
    sin_theta = jnp.linalg.norm(rho, axis=-1) / 2
    cos_theta = jnp.clip((r00 + r11 + r22 - 1) / 2, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)
    near_pi = sin_theta < _NEAR_PI_SIN_THRESH
    rvec = rho * (theta / jnp.where(near_pi, 1.0, 2.0 * sin_theta))[..., None]
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
    axis_norm = jnp.linalg.norm(axis, axis=-1)
    rvec_pi = axis * (theta / jnp.where(axis_norm == 0, 1.0, axis_norm))[..., None]
    rvec_degenerate = jnp.where((cos_theta > 0)[..., None], 0.0, rvec_pi)
    return jnp.where(near_pi[..., None], rvec_degenerate, rvec)


def distort_manual(pts2d, dists):
    n = dists.shape[-1]
    if n == 0:
        return pts2d
    coef_shape = (dists.shape[0],) + (1,) * (pts2d.ndim - 2)
    k = [dists[:, i].reshape(coef_shape) for i in range(n)]
    x, y = pts2d[..., 0], pts2d[..., 1]
    x2, y2 = x * x, y * y
    r2 = x2 + y2
    num = 1.0 + k[0] * r2
    r4 = r6 = None
    if n >= 2:
        r4 = r2 * r2
        num = num + k[1] * r4
    if n >= 5:
        r6 = r4 * r2
        num = num + k[4] * r6
    den = 1.0
    if n >= 6:
        den = den + k[5] * r2
    if n >= 7:
        den = den + k[6] * r4
    if n >= 8:
        den = den + k[7] * r6
    mult = num / den
    add_x = jnp.zeros_like(x)
    add_y = jnp.zeros_like(y)
    if n >= 3:
        xy = x * y
        add_x = 2 * k[2] * xy
        add_y = k[2] * (r2 + 2 * y2)
    if n >= 4:
        add_x = add_x + k[3] * (r2 + 2 * x2)
        add_y = add_y + 2 * k[3] * xy
    if n >= 9:
        add_x = add_x + k[8] * r2
    if n >= 10:
        add_x = add_x + k[9] * r4
    if n >= 11:
        add_y = add_y + k[10] * r2
    if n >= 12:
        add_y = add_y + k[11] * r4
    return jnp.stack([x * mult + add_x, y * mult + add_y], axis=-1)


def project_full_manual(pts3d, rvecs, tvecs, intrs, dists):
    pts_dims = pts3d.shape[:-1]
    rmats = rvec_to_rmat_manual(rvecs)
    xyz = ((rmats @ pts3d.T) + tvecs[..., None]).transpose(0, 2, 1)
    xy = xyz[:, :, :2] / xyz[:, :, 2:3]
    xy = distort_manual(xy, dists)
    intr_shape = (*intrs.shape[:-1], *(1,) * len(pts_dims), 2)
    return xy * intrs[..., jnp.array([0, -3])].reshape(intr_shape) + intrs[
        ..., -2:
    ].reshape(intr_shape)


# -- jit ---------------------------------------------------------------------

rvec_to_rmat_manual_jit = jax.jit(rvec_to_rmat_manual)
rvec_to_rmat_jit = jax.jit(rvec_to_rmat)
rmat_to_rvec_manual_jit = jax.jit(rmat_to_rvec_manual)
rmat_to_rvec_jit = jax.jit(rmat_to_rvec)
distort_manual_jit = jax.jit(distort_manual)
distort_jit = jax.jit(distort)
project_full_manual_jit = jax.jit(project_full_manual)
project_full_jit = jax.jit(project_full)


# -- inputs ------------------------------------------------------------------

rng = np.random.default_rng(0)
rvecs = jnp.asarray(rng.normal(size=(V, 3)) * 0.3)
rmats = rvec_to_rmat(rvecs)
tvecs = jnp.asarray(rng.normal(size=(V, 3)))
intrs = jnp.asarray(np.tile(np.array([800.0, 800.0, 320.0, 240.0]), (V, 1)))
pts3d = jnp.asarray(rng.normal(size=(N, 3)) + np.array([0.0, 0.0, 5.0]))
dists = jnp.asarray(rng.normal(size=(V, 5)) * 0.01)
pts2d = jnp.asarray(rng.normal(size=(V, N, 2)))


# -- bench helper ------------------------------------------------------------


def bench(fn, *args):
    for _ in range(N_WARMUP):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) / N_REPEAT


def check_close(name: str, a, b, atol: float = 1e-10):
    diff = float(jnp.max(jnp.abs(a - b)))
    status = "ok" if diff < atol else "FAIL"
    print(f"  {name:<16} max|Δ| = {diff:.2e}  [{status}]")


# -- run ---------------------------------------------------------------------

print(f"Shape: V={V}, N={N}")
print()
print("Correctness (vmap library impl vs old manual impl):")
check_close("rvec_to_rmat", rvec_to_rmat(rvecs), rvec_to_rmat_manual(rvecs))
check_close("rmat_to_rvec", rmat_to_rvec(rmats), rmat_to_rvec_manual(rmats))
check_close("distort", distort(pts2d, dists), distort_manual(pts2d, dists))
check_close(
    "project_full",
    project_full(pts3d, rvecs, tvecs, intrs, dists),
    project_full_manual(pts3d, rvecs, tvecs, intrs, dists),
)

print()
print(f"{'function':<16} {'manual (µs)':>12} {'vmap (µs)':>12} {'ratio':>8}")
print("-" * 52)

cases = [
    ("rvec_to_rmat", rvec_to_rmat_manual_jit, rvec_to_rmat_jit, (rvecs,)),
    ("rmat_to_rvec", rmat_to_rvec_manual_jit, rmat_to_rvec_jit, (rmats,)),
    ("distort", distort_manual_jit, distort_jit, (pts2d, dists)),
    (
        "project_full",
        project_full_manual_jit,
        project_full_jit,
        (pts3d, rvecs, tvecs, intrs, dists),
    ),
]

for name, manual_fn, vmap_fn, args in cases:
    t_manual = bench(manual_fn, *args)
    t_vmap = bench(vmap_fn, *args)
    print(
        f"{name:<16} {t_manual * 1e6:>12.2f} {t_vmap * 1e6:>12.2f} "
        f"{t_vmap / t_manual:>7.2f}x"
    )
