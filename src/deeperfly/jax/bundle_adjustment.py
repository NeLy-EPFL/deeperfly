"""Bundle adjustment with JAX-accelerated residuals and Jacobians.

Two solvers are provided:

- :func:`bundle_adjust` wraps :func:`scipy.optimize.least_squares` (TRF +
  LSMR). The per-observation residual and its Jacobian are computed via
  :func:`jax.vmap` + :func:`jax.jacfwd` on :func:`project_one`, then re-assembled
  into a sparse SciPy matrix using the precomputed sparsity pattern.
- :func:`bundle_adjust_optx` runs :mod:`optimistix` Levenberg--Marquardt
  entirely inside JAX, using either matrix-free LSMR, matrix-free CG on the
  normal equations, or a dense QR factorisation as the linear solver.

The packed-state convention (``values`` + ``fixed`` + ``*_idx`` arrays +
``pts2d``) is defined in :mod:`deeperfly.ba_state`; build it with that
module's :func:`prep_args`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
from jaxtyping import Array, Bool, Float, Int
from scipy.optimize import OptimizeResult, least_squares
from scipy.sparse import csr_matrix

from .geometry import project_full_one

jax.config.update("jax_enable_x64", True)


# Per-observation projection and its Jacobian w.r.t. all five parameter groups
# (pt3d, rvec, tvec, intr, dist). The Jacobian tuple is returned in the order
# of project_one's arguments and we keep cols_per_obs aligned with it below.
_project_per_obs = jax.jit(jax.vmap(project_full_one))
_jac_per_obs = jax.jit(jax.vmap(jax.jacfwd(project_full_one, argnums=(0, 1, 2, 3, 4))))


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
    """Bundle adjustment with a JAX-computed analytic Jacobian.

    Drop-in replacement for :func:`deeperfly.bundle_adjustment.bundle_adjust`:
    same packed-state interface, same return shape.

    Parameters
    ----------
    values, fixed, rvecs_idx, tvecs_idx, intrs_idx, dists_idx, pts3d_idx, pts2d
        See :class:`deeperfly.ba_state.BAState`.
    loss, f_scale, max_nfev, **kwargs
        Forwarded to :func:`scipy.optimize.least_squares`.

    Returns
    -------
    ``(result, (rvecs, tvecs, intrs, dists, pts3d))``.
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

    def residuals(x):
        rvecs, tvecs, intrs, dists, pts3d = unpack(x)
        pts2d_predicted = _project_obs(rvecs, tvecs, intrs, dists, pts3d)
        return (np.asarray(pts2d_predicted) - pts2d_observed).ravel()

    # Sparsity pattern. Each observation's two residual rows depend on:
    # the 3D point's slot plus its view's rvec / tvec / intr / dist slots.
    # The column order MUST match the order of the Jacobian tuple returned
    # by ``_jac_per_obs`` (which is the order of project_one's arguments).
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
        # axis to align with cols_per_obs above.
        jac_full = np.concatenate([np.asarray(j) for j in jac_blocks], axis=-1)
        data = np.concatenate([jac_full[:, 0, :][valid], jac_full[:, 1, :][valid]])
        return csr_matrix((data, (rows, cols)), shape=(2 * n_obs, n_free))

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
    return result, unpack(result.x)


# -- Optimistix solver (pure-JAX Levenberg--Marquardt) -----------------------


_project_per_obs_vmap = jax.vmap(project_full_one)


def _optx_residuals(x: Float[Array, "n_free"], args: tuple) -> Float[Array, "2*n_obs"]:
    """Pure-JAX reprojection residuals for Optimistix.

    Module-level (rather than a closure) so JAX's trace cache is reused
    across repeated calls with the same shapes.
    """
    (
        values,
        free_idx,
        rvecs_idx,
        tvecs_idx,
        intrs_idx,
        dists_idx,
        pts3d_idx,
        obs_view,
        obs_pt,
        pts2d_observed,
    ) = args
    full = values.at[free_idx].set(x)
    rvecs = full[rvecs_idx]
    tvecs = full[tvecs_idx]
    intrs = full[intrs_idx]
    dists = full[dists_idx]
    pts3d = full[pts3d_idx]
    pts2d_predicted = _project_per_obs_vmap(
        pts3d[obs_pt],
        rvecs[obs_view],
        tvecs[obs_view],
        intrs[obs_view],
        dists[obs_view],
    )
    return (pts2d_predicted - pts2d_observed).ravel()


def bundle_adjust_optx(
    values: Float[np.ndarray, "n_params"],
    fixed: Bool[np.ndarray, "n_params"],
    rvecs_idx: Int[np.ndarray, "V 3"],
    tvecs_idx: Int[np.ndarray, "V 3"],
    intrs_idx: Int[np.ndarray, "V P"],
    dists_idx: Int[np.ndarray, "V K"],
    pts3d_idx: Int[np.ndarray, "N 3"],
    pts2d: Float[np.ndarray, "V N 2"],
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 1000,
    linear_solver: str = "lsmr",
    lsmr_rtol: float = 1e-6,
    lsmr_atol: float = 1e-6,
    verbose: bool = False,
) -> tuple[
    optx.Solution, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """Bundle adjustment via Optimistix Levenberg--Marquardt.

    Same packed-state interface as :func:`bundle_adjust`.

    Parameters
    ----------
    values, fixed, rvecs_idx, tvecs_idx, intrs_idx, dists_idx, pts3d_idx, pts2d
        See :class:`deeperfly.ba_state.BAState`.
    rtol, atol, max_steps, verbose
        Optimistix LM termination criteria.
    linear_solver
        Linear solver used at each LM step:

        - ``"lsmr"`` (default): matrix-free LSMR via JVPs (sparsity-friendly).
        - ``"normal_cg"``: matrix-free CG on the normal equations.
        - ``"qr"``: dense QR factorisation (Optimistix default).
    lsmr_rtol, lsmr_atol
        Tolerances for LSMR / NormalCG (ignored for QR).

    Returns
    -------
    ``(solution, (rvecs, tvecs, intrs, dists, pts3d))`` where ``solution`` is
    the Optimistix solution object.
    """
    obs_view, obs_pt = np.where(np.isfinite(pts2d).all(axis=-1))
    free_idx = np.where(~fixed)[0]

    args = (
        jnp.asarray(values),
        jnp.asarray(free_idx),
        jnp.asarray(rvecs_idx),
        jnp.asarray(tvecs_idx),
        jnp.asarray(intrs_idx),
        jnp.asarray(dists_idx),
        jnp.asarray(pts3d_idx),
        jnp.asarray(obs_view),
        jnp.asarray(obs_pt),
        jnp.asarray(pts2d[obs_view, obs_pt]),
    )

    if linear_solver == "lsmr":
        linear = lx.LSMR(rtol=lsmr_rtol, atol=lsmr_atol)
    elif linear_solver == "normal_cg":
        linear = lx.NormalCG(rtol=lsmr_rtol, atol=lsmr_atol)
    elif linear_solver == "qr":
        linear = lx.QR()
    else:
        raise ValueError(f"unknown linear_solver: {linear_solver!r}")

    solver = optx.LevenbergMarquardt(
        rtol=rtol,
        atol=atol,
        linear_solver=linear,
        verbose=verbose,
    )
    x0 = args[0][args[1]]
    solution = optx.least_squares(
        _optx_residuals,
        solver,
        x0,
        args=args,
        max_steps=max_steps,
        throw=False,
    )

    full_opt = args[0].at[args[1]].set(solution.value)
    unpacked = tuple(
        np.asarray(full_opt[idx])
        for idx in (args[2], args[3], args[4], args[5], args[6])
    )
    return solution, unpacked
