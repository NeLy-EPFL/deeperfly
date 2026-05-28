"""Sparse bundle adjustment via :func:`scipy.optimize.least_squares`.

The numeric Jacobian is filled in by :mod:`scipy` using the sparsity pattern
built once up front -- each residual (one observation, two rows) depends only
on the parameters of its view and the coordinates of its 3D point.

The packed-state convention (``values`` + ``fixed`` + ``*_idx`` arrays +
``pts2d``) is defined in :mod:`deeperfly.ba_state`; build it with that
module's :func:`prep_args`.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Bool, Float, Int
from scipy.optimize import OptimizeResult, least_squares
from scipy.sparse import csr_matrix

from .geometry import project_full


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
    **kwargs,
) -> tuple[
    OptimizeResult, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """Run sparse bundle adjustment.

    Minimises the sum of squared reprojection residuals over the free entries
    of ``values`` using :func:`scipy.optimize.least_squares` (Trust Region
    Reflective + LSMR). The Jacobian sparsity pattern is precomputed: each
    observation contributes two rows that depend only on its view's camera
    parameters and the coordinates of the observed 3D point.

    Parameters
    ----------
    values, fixed, rvecs_idx, tvecs_idx, intrs_idx, dists_idx, pts3d_idx, pts2d
        See :class:`deeperfly.ba_state.BAState`.
    loss, f_scale, max_nfev, **kwargs
        Forwarded to :func:`scipy.optimize.least_squares`.

    Returns
    -------
    ``(result, (rvecs, tvecs, intrs, dists, pts3d))`` where ``result`` is the
    scipy result object and the second element holds the unpacked optimised
    arrays.
    """
    obs_view, obs_pt = np.where(np.isfinite(pts2d).all(axis=-1))
    pts2d_observed = pts2d[obs_view, obs_pt]
    n_obs = len(obs_view)
    n_free = int((~fixed).sum())

    x0 = values[~fixed]
    rvecs = values[rvecs_idx]
    tvecs = values[tvecs_idx]
    intrs = values[intrs_idx]
    dists = values[dists_idx]
    pts3d = values[pts3d_idx]

    # For each named slot, build an (assign-mask, source-index) pair so we can
    # scatter the optimiser's free vector ``x`` back into the unpacked arrays
    # without touching fixed entries.
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

    def residuals(x):
        rvecs, tvecs, intrs, dists, pts3d = unpack(x)
        pts2d_predicted = project_full(pts3d, rvecs, tvecs, intrs, dists)
        return (pts2d_predicted[obs_view, obs_pt] - pts2d_observed).ravel()

    # Jacobian sparsity. Each observation's two residual rows depend on:
    # its view's rvec/tvec/intr/dist slots plus its point's pts3d slot.
    # We collect those column indices per observation, then drop the fixed ones.
    free_idx_map = np.full(values.size, -1, dtype=np.int64)
    free_idx_map[~fixed] = np.arange(n_free)
    cols_per_obs = np.concatenate(
        [
            rvecs_idx[obs_view],
            tvecs_idx[obs_view],
            intrs_idx[obs_view],
            dists_idx[obs_view],
            pts3d_idx[obs_pt],
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
    jac_sparsity = csr_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)),
        shape=(2 * n_obs, n_free),
    )

    kwargs.setdefault("x_scale", "jac")
    result = least_squares(
        residuals,
        x0,
        jac_sparsity=jac_sparsity,
        method="trf",
        tr_solver="lsmr",
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        **kwargs,
    )
    return result, unpack(result.x)
