"""Bundle adjustment with JAX-accelerated residuals and Jacobians.

- :func:`bundle_adjust` wraps :func:`scipy.optimize.least_squares` (TRF +
  LSMR). The per-observation residual and its Jacobian are computed via
  :func:`jax.vmap` + :func:`jax.jacfwd` on
  :func:`deeperfly.geometry.project_full_one`, then re-assembled into a sparse
  SciPy matrix using the precomputed sparsity pattern.

  The packed-state convention (``values`` + ``fixed`` + ``*_idx`` arrays +
``pts2d``) is defined in :mod:`deeperfly.bundle_adjustment.state`; build it with
that module's :func:`build_state`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Bool, Float, Int
from scipy.optimize import OptimizeResult, least_squares
from scipy.sparse import csr_matrix

from ..geometry import project_full_one

jax.config.update("jax_enable_x64", True)


class BASolution(NamedTuple):
    """Optimized parameters unpacked back into per-camera arrays plus points."""

    rvecs: np.ndarray
    tvecs: np.ndarray
    intrs: np.ndarray
    dists: np.ndarray
    pts3d: np.ndarray


# Per-observation projection and its Jacobian w.r.t. all five parameter groups
# (pt3d, rvec, tvec, intr, dist). The Jacobian tuple is returned in the order
# of project_full_one's arguments and we keep cols_per_obs aligned with it below.
_project_per_obs = jax.jit(jax.vmap(project_full_one))
_jac_per_obs = jax.jit(jax.vmap(jax.jacfwd(project_full_one, argnums=(0, 1, 2, 3, 4))))


def _bone_length_one(pi: Float[jnp.ndarray, "3"], pj: Float[jnp.ndarray, "3"]):
    """Euclidean distance between two 3D points (per-bone primitive)."""
    return jnp.linalg.norm(pi - pj)


# Bone length and its Jacobian w.r.t. each endpoint, vmapped over bones.
_bone_len_per = jax.jit(jax.vmap(_bone_length_one))
_bone_jac_per = jax.jit(jax.vmap(jax.jacfwd(_bone_length_one, argnums=(0, 1))))


def bundle_adjust(
    values: Float[np.ndarray, "n_params"],
    fixed: Bool[np.ndarray, "n_params"],
    rvecs_idx: Int[np.ndarray, "V 3"],
    tvecs_idx: Int[np.ndarray, "V 3"],
    intrs_idx: Int[np.ndarray, "V P"],
    dists_idx: Int[np.ndarray, "V K"],
    pts3d_idx: Int[np.ndarray, "N 3"],
    pts2d: Float[np.ndarray, "V N 2"],
    loss: str = "linear",
    f_scale: float = 1.0,
    max_nfev: int = 1000,
    weights: Float[np.ndarray, "V N"] | None = None,
    bone_pairs: Int[np.ndarray, "B 2"] | None = None,
    bone_targets: Float[np.ndarray, "B"] | None = None,
    bone_weight: float = 1.0,
    **kwargs,
) -> tuple[OptimizeResult, BASolution]:
    """Bundle adjustment with a JAX-computed analytic Jacobian.

    Low-level solver over the packed state. For the config-driven, camera-aware
    entry point see :func:`deeperfly.bundle_adjustment.bundle_adjust`.

    Parameters
    ----------
    values, fixed, rvecs_idx, tvecs_idx, intrs_idx, dists_idx, pts3d_idx, pts2d
        See :class:`deeperfly.bundle_adjustment.state.BAState`.
    loss, f_scale, max_nfev, **kwargs
        Forwarded to :func:`scipy.optimize.least_squares`. Use ``loss="huber"``
        with ``f_scale`` set to a pixel threshold for robust calibration.
    weights
        Optional per-observation weights of shape ``(V, N)``. Each reprojection
        residual is scaled by ``sqrt(weight)`` (so the cost is ``weight``
        times the squared pixel error) -- pass detector confidences here.
    bone_pairs
        Optional ``(B, 2)`` point-index pairs adding a soft bone-length prior:
        for each pair a residual ``bone_weight * (||p_i - p_j|| - target)`` is
        appended. Indices refer to rows of the points array (``pts3d_idx``).
    bone_targets
        Target lengths of shape ``(B,)`` for ``bone_pairs`` (required when
        ``bone_pairs`` is given).
    bone_weight
        Scalar weight on the bone-length residuals.

    Returns
    -------
    ``(result, BASolution(rvecs, tvecs, intrs, dists, pts3d))``.
    """
    obs_view, obs_pt = np.where(np.isfinite(pts2d).all(axis=-1))
    pts2d_observed = pts2d[obs_view, obs_pt]
    n_obs = len(obs_view)
    n_free = int((~fixed).sum())

    if weights is None:
        sqrt_w = np.ones(n_obs)
    else:
        sqrt_w = np.sqrt(np.asarray(weights, dtype=float)[obs_view, obs_pt])

    use_bones = bone_pairs is not None and len(bone_pairs) > 0
    if use_bones:
        if bone_targets is None:
            raise ValueError("bone_targets is required when bone_pairs is given")
        bone_pairs = np.asarray(bone_pairs)
        bone_targets = np.asarray(bone_targets, dtype=float)
        bone_i, bone_j = bone_pairs[:, 0], bone_pairs[:, 1]
        n_bones = len(bone_pairs)
    else:
        n_bones = 0

    x0 = values[~fixed]
    rvecs = values[rvecs_idx]
    tvecs = values[tvecs_idx]
    intrs = values[intrs_idx]
    dists = values[dists_idx]
    pts3d = values[pts3d_idx]

    free_cumsum = np.cumsum(~fixed)

    def free_assign(idx):
        assign_mask = ~fixed[idx]
        source_idx = free_cumsum[idx][assign_mask] - 1
        return assign_mask, source_idx

    rvecs_assign = free_assign(rvecs_idx)
    tvecs_assign = free_assign(tvecs_idx)
    intrs_assign = free_assign(intrs_idx)
    dists_assign = free_assign(dists_idx)
    pts3d_assign = free_assign(pts3d_idx)

    def unpack(x):
        rvecs[rvecs_assign[0]] = x[rvecs_assign[1]]
        tvecs[tvecs_assign[0]] = x[tvecs_assign[1]]
        intrs[intrs_assign[0]] = x[intrs_assign[1]]
        dists[dists_assign[0]] = x[dists_assign[1]]
        pts3d[pts3d_assign[0]] = x[pts3d_assign[1]]
        return rvecs, tvecs, intrs, dists, pts3d

    def _project_obs(rvecs, tvecs, intrs, dists, pts3d):
        return _project_per_obs(
            jnp.asarray(pts3d[obs_pt]),
            jnp.asarray(rvecs[obs_view]),
            jnp.asarray(tvecs[obs_view]),
            jnp.asarray(intrs[obs_view]),
            jnp.asarray(dists[obs_view]),
        )

    n_resid = 2 * n_obs + n_bones

    def residuals(x):
        rvecs, tvecs, intrs, dists, pts3d = unpack(x)
        pts2d_predicted = _project_obs(rvecs, tvecs, intrs, dists, pts3d)
        reproj = (
            (np.asarray(pts2d_predicted) - pts2d_observed) * sqrt_w[:, None]
        ).ravel()
        if not use_bones:
            return reproj
        lengths = np.asarray(
            _bone_len_per(jnp.asarray(pts3d[bone_i]), jnp.asarray(pts3d[bone_j]))
        )
        bone_resid = bone_weight * (lengths - bone_targets)
        return np.concatenate([reproj, bone_resid])

    # Sparsity pattern. Each observation's two residual rows depend on:
    # the 3D point's slot plus its view's rvec / tvec / intr / dist slots.
    # The column order MUST match the order of the Jacobian tuple returned
    # by ``_jac_per_obs`` (the order of project_full_one's arguments).
    free_idx_map = np.full(values.size, -1, dtype=np.int64)
    free_idx_map[~fixed] = np.arange(n_free)
    cols_per_obs = np.concatenate(
        [
            pts3d_idx[obs_pt],
            rvecs_idx[obs_view],
            tvecs_idx[obs_view],
            intrs_idx[obs_view],
            dists_idx[obs_view],
        ],
        axis=1,
    )
    free_cols_per_obs = free_idx_map[cols_per_obs]
    valid = free_cols_per_obs >= 0
    cols_nz = free_cols_per_obs[valid]
    obs_idx = np.broadcast_to(np.arange(n_obs)[:, None], cols_per_obs.shape)
    rows_x = 2 * obs_idx[valid]
    rows = np.concatenate([rows_x, rows_x + 1])
    cols = np.concatenate([cols_nz, cols_nz])

    # Bone-length sparsity. Each bone row b depends only on the 3 coordinate
    # slots of its two endpoints; its row index sits after the reprojection rows.
    if use_bones:
        bone_cols = np.concatenate(
            [pts3d_idx[bone_i], pts3d_idx[bone_j]], axis=1
        )  # (B,6)
        free_bone_cols = free_idx_map[bone_cols]
        bone_valid = free_bone_cols >= 0
        bone_rows_all = np.broadcast_to(
            (2 * n_obs + np.arange(n_bones))[:, None], bone_cols.shape
        )
        bone_rows = bone_rows_all[bone_valid]
        bone_cols_nz = free_bone_cols[bone_valid]
        rows = np.concatenate([rows, bone_rows])
        cols = np.concatenate([cols, bone_cols_nz])

    def jac(x):
        rvecs, tvecs, intrs, dists, pts3d = unpack(x)
        jac_blocks = _jac_per_obs(
            jnp.asarray(pts3d[obs_pt]),
            jnp.asarray(rvecs[obs_view]),
            jnp.asarray(tvecs[obs_view]),
            jnp.asarray(intrs[obs_view]),
            jnp.asarray(dists[obs_view]),
        )
        # Each block is (n_obs, 2, *param_dim); concatenate along the last
        # axis to align with cols_per_obs above. Weight rows by sqrt(weight).
        jac_full = np.concatenate([np.asarray(j) for j in jac_blocks], axis=-1)
        jac_full = jac_full * sqrt_w[:, None, None]
        data = np.concatenate([jac_full[:, 0, :][valid], jac_full[:, 1, :][valid]])
        if use_bones:
            dpi, dpj = _bone_jac_per(
                jnp.asarray(pts3d[bone_i]), jnp.asarray(pts3d[bone_j])
            )
            bone_jac = bone_weight * np.concatenate(
                [np.asarray(dpi), np.asarray(dpj)], axis=1
            )  # (B, 6)
            data = np.concatenate([data, bone_jac[bone_valid]])
        return csr_matrix((data, (rows, cols)), shape=(n_resid, n_free))

    result = least_squares(
        residuals,
        x0,
        jac=jac,
        method="trf",
        tr_solver="lsmr",
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        **kwargs,
    )

    solution = BASolution(*unpack(result.x))
    return result, solution
