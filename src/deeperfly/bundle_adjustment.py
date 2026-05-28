"""Sparse bundle adjustment via :func:`scipy.optimize.least_squares`.

The state vector concatenates ``[rvecs, tvecs, intrs, dists, pts3d]`` flat.
Index arrays (``rvecs_idx``, etc.) recover the original shapes; a boolean
``fixed`` mask marks parameters that are held constant during optimisation.
``intrs`` and ``dists`` are broadcast across views by default (shared camera
model), but the indexing scheme is general enough to allow per-view freedom by
flipping entries in ``fixed``.

The numeric Jacobian is filled in by :mod:`scipy` using the sparsity pattern
built once up front -- each residual (one observation, two rows) depends only
on the parameters of its view and the coordinates of its 3D point.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from jaxtyping import Bool, Float, Int
from scipy.optimize import OptimizeResult, least_squares
from scipy.sparse import csr_matrix

from .multiview_geom import (
    intr_to_kmat,
    project_full,
    rvec_to_rmat,
    triangulate_dlt,
)


class BAState(NamedTuple):
    """Packed bundle-adjustment state ready for :func:`bundle_adjust`.

    ``values`` is the flat parameter vector; ``fixed`` is a boolean mask of
    parameters to hold constant. The ``*_idx`` arrays index into ``values`` to
    recover the original ``rvecs`` / ``tvecs`` / ``intrs`` / ``dists`` /
    ``pts3d`` arrays. ``pts2d`` is the observation tensor.
    """

    values: Float[np.ndarray, "n_params"]
    fixed: Bool[np.ndarray, "n_params"]
    rvecs_idx: Int[np.ndarray, "V 3"]
    tvecs_idx: Int[np.ndarray, "V 3"]
    intrs_idx: Int[np.ndarray, "V P"]
    dists_idx: Int[np.ndarray, "V K"]
    pts3d_idx: Int[np.ndarray, "N 3"]
    pts2d: Float[np.ndarray, "V N 2"]


def initialize_pts3d(
    pts2d: Float[np.ndarray, "V *pts 2"],
    rvecs: Float[np.ndarray, "V 3"],
    tvecs: Float[np.ndarray, "V 3"],
    intrs: Float[np.ndarray, "V P"] | Float[np.ndarray, "P"],
) -> Float[np.ndarray, "*pts 3"]:
    """Triangulate initial 3D points from 2D observations and camera poses."""
    kmat = intr_to_kmat(intrs)
    rtmat = np.concatenate((rvec_to_rmat(rvecs), tvecs[..., None]), axis=-1)
    return triangulate_dlt(pts2d, kmat @ rtmat)


def prep_args(
    pts2d: Float[np.ndarray, "V N 2"],
    rvecs: Float[np.ndarray, "V 3"],
    tvecs: Float[np.ndarray, "V 3"],
    intrs: Float[np.ndarray, "V P"] | Float[np.ndarray, "P"],
    dists: Float[np.ndarray, "V K"] | Float[np.ndarray, "K"],
) -> BAState:
    """Build a :class:`BAState` from observations and initial camera parameters.

    The 3D points are initialised by triangulating ``pts2d`` from the given
    cameras. Intrinsics and distortion are broadcast to every view and then
    marked fixed in the returned ``fixed`` mask, so the optimiser solves only
    for extrinsics and 3D points by default. Flip the corresponding entries of
    ``fixed`` to free them.

    Parameters
    ----------
    pts2d
        Observed 2D points of shape ``(V, N, 2)`` with NaNs for missing.
    rvecs, tvecs
        Initial per-view extrinsics of shape ``(V, 3)``.
    intrs, dists
        Initial intrinsics / distortion, shared across views.

    Returns
    -------
    A :class:`BAState` ready to splat into :func:`bundle_adjust`.
    """
    pts3d = initialize_pts3d(pts2d, rvecs, tvecs, intrs)
    arrs = [rvecs, tvecs, intrs, dists, pts3d]
    values = np.concatenate([a.ravel() for a in arrs])
    size_cumsum = [0, *np.cumsum([a.size for a in arrs])]
    fixed = np.zeros_like(values, dtype=bool)
    # Fix intrinsics and distortions by default.
    fixed[size_cumsum[2] : size_cumsum[4]] = True
    rvecs_idx, tvecs_idx, intrs_idx, dists_idx, pts3d_idx = (
        np.arange(a.size).reshape(a.shape) + size_cumsum[i] for i, a in enumerate(arrs)
    )
    # Broadcast shared intrinsics / distortions to all V cameras.
    n_views = len(rvecs)
    intrs_idx = np.stack((intrs_idx,) * n_views, axis=0)
    dists_idx = np.stack((dists_idx,) * n_views, axis=0)
    return BAState(
        values, fixed, rvecs_idx, tvecs_idx, intrs_idx, dists_idx, pts3d_idx, pts2d
    )


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
    values
        Flat parameter vector (mutated copies are returned in the second
        element of the result tuple).
    fixed
        Boolean mask of entries in ``values`` held constant during optimisation.
    rvecs_idx, tvecs_idx, intrs_idx, dists_idx, pts3d_idx
        Index arrays into ``values`` (i.e. ``values[rvecs_idx]`` recovers the
        per-view rotation vectors with their original shape, etc.).
    pts2d
        2D observations of shape ``(V, N, 2)``; NaNs are skipped.
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
