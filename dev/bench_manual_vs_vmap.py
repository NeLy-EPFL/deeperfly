"""Benchmark manual batched JAX impls vs vmap of *_one siblings.

Compares ``rvec_to_rmat``, ``distort``, ``project_full`` in
``deeperfly.jax.geometry`` against equivalent ``vmap``-of-``*_one``
versions, under ``jax.jit``, at the typical BA shape (V=7, N=200).

Run as ``uv run python dev/bench_manual_vs_vmap.py``.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from deeperfly.jax.geometry import (
    distort,
    distort_one,
    project_full,
    project_one,
    rvec_to_rmat,
    rvec_to_rmat_one,
)

jax.config.update("jax_enable_x64", True)

V, N = 7, 2000
N_REPEAT = 500
N_WARMUP = 10


# -- vmap-based replacements -------------------------------------------------

rvec_to_rmat_vmap = jax.vmap(rvec_to_rmat_one)

distort_vmap = jax.vmap(
    jax.vmap(distort_one, in_axes=(0, None)),
    in_axes=(0, 0),
)

project_full_vmap = jax.vmap(
    jax.vmap(project_one, in_axes=(0, None, None, None, None)),
    in_axes=(None, 0, 0, 0, 0),
)


# -- jit ---------------------------------------------------------------------

rvec_to_rmat_jit = jax.jit(rvec_to_rmat)
rvec_to_rmat_vmap_jit = jax.jit(rvec_to_rmat_vmap)
distort_jit = jax.jit(distort)
distort_vmap_jit = jax.jit(distort_vmap)
project_full_jit = jax.jit(project_full)
project_full_vmap_jit = jax.jit(project_full_vmap)


# -- inputs ------------------------------------------------------------------

rng = np.random.default_rng(0)
rvecs = jnp.asarray(rng.normal(size=(V, 3)) * 0.3)
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
print("Correctness:")
check_close("rvec_to_rmat", rvec_to_rmat(rvecs), rvec_to_rmat_vmap(rvecs))
check_close("distort", distort(pts2d, dists), distort_vmap(pts2d, dists))
check_close(
    "project_full",
    project_full(pts3d, rvecs, tvecs, intrs, dists),
    project_full_vmap(pts3d, rvecs, tvecs, intrs, dists),
)

print()
print(f"{'function':<16} {'manual (µs)':>12} {'vmap (µs)':>12} {'ratio':>8}")
print("-" * 52)

cases = [
    ("rvec_to_rmat", rvec_to_rmat_jit, rvec_to_rmat_vmap_jit, (rvecs,)),
    ("distort", distort_jit, distort_vmap_jit, (pts2d, dists)),
    (
        "project_full",
        project_full_jit,
        project_full_vmap_jit,
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
