import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from .multiview_geom import intr2kmat, rvec2mat, triangulate_dlt
from .multiview_geom_jax import project_one

jax.config.update("jax_enable_x64", True)


def initialize_pts3d(
    pts2d: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    intrs: np.ndarray,
):
    kmat = intr2kmat(intrs)
    rtmat = np.concatenate((rvec2mat(rvecs), tvecs[..., None]), axis=-1)
    return triangulate_dlt(kmat @ rtmat, pts2d)


def prep_args(
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    intrs: np.ndarray,
    dists: np.ndarray,
    pts2d: np.ndarray,
):
    pts3d = initialize_pts3d(pts2d, rvecs, tvecs, intrs)
    arrs = [rvecs, tvecs, intrs, dists, pts3d]
    values = np.concatenate([a.ravel() for a in arrs])
    size_cumsum = [0, *np.cumsum([a.size for a in arrs])]
    fixed = np.zeros_like(values, dtype=bool)
    fixed[size_cumsum[2] : size_cumsum[4]] = True  # fix intrinsics and distortions
    rvecs_idx, tvecs_idx, intrs_idx, dists_idx, pts3d_idx = (
        np.arange(a.size).reshape(a.shape) + size_cumsum[i] for i, a in enumerate(arrs)
    )
    intrs_idx = np.stack(
        (intrs_idx,) * len(rvecs), axis=0
    )  # broadcast intrinsics to all cameras
    dists_idx = np.stack(
        (dists_idx,) * len(rvecs), axis=0
    )  # broadcast distortions to all cameras
    return values, fixed, rvecs_idx, tvecs_idx, intrs_idx, dists_idx, pts3d_idx, pts2d


_project_per_obs = jax.jit(jax.vmap(project_one))
_jac_per_obs = jax.jit(jax.vmap(jax.jacfwd(project_one, argnums=(0, 1, 2, 3, 4))))


def bundle_adjust(
    values: np.ndarray,
    fixed: np.ndarray,
    rvecs_idx: np.ndarray,
    tvecs_idx: np.ndarray,
    intrs_idx: np.ndarray,
    dists_idx: np.ndarray,
    pts3d_idx: np.ndarray,
    pts2d: np.ndarray,
    loss="linear",
    f_scale=1.0,
    max_nfev=1000,
    **kwargs,
):
    """Bundle adjustment

    Parameters
    ----------
    values : np.ndarray
        1D array containing values of parameters
    fixed : np.ndarray
        Boolean mask indicating which values are fixed (not optimized)
    rvecs_idx : np.ndarray
        Indices of rotation vectors in `values` (i.e., values[rvecs_idx] gives the rotation vectors)
    tvecs_idx : np.ndarray
        Indices of translation vectors in `values` (i.e., values[tvecs_idx] gives the translation vectors)
    intrs_idx : np.ndarray
        Indices of intrinsic parameters in `values` (i.e., values[intrs_idx] gives the intrinsic parameters)
    dists_idx : np.ndarray
        Indices of distortion parameters in `values` (i.e., values[dists_idx] gives the distortion parameters)
    pts3d_idx : np.ndarray
        Indices of 3D points in `values` (i.e., values[pts3d_idx] gives the 3D points)
    pts2d : np.ndarray
        2D points (observations)

    """

    def get_lr_idx(idx):
        l_idx = ~fixed[idx]
        r_idx = np.cumsum(~fixed)[idx][l_idx] - 1
        return l_idx, r_idx

    obs_v, obs_n = np.where(np.isfinite(pts2d).all(axis=-1))
    pts2d_obs = pts2d[obs_v, obs_n]
    n_obs = len(obs_v)
    n_free = int((~fixed).sum())

    x = values[~fixed]
    rvecs = values[rvecs_idx]
    tvecs = values[tvecs_idx]
    intrs = values[intrs_idx]
    dists = values[dists_idx]
    pts3d = values[pts3d_idx]
    rvecs_idx_l, rvecs_idx_r = get_lr_idx(rvecs_idx)
    tvecs_idx_l, tvecs_idx_r = get_lr_idx(tvecs_idx)
    intrs_idx_l, intrs_idx_r = get_lr_idx(intrs_idx)
    dists_idx_l, dists_idx_r = get_lr_idx(dists_idx)
    pts3d_idx_l, pts3d_idx_r = get_lr_idx(pts3d_idx)

    def unpack(x):
        rvecs[rvecs_idx_l] = x[rvecs_idx_r]
        tvecs[tvecs_idx_l] = x[tvecs_idx_r]
        intrs[intrs_idx_l] = x[intrs_idx_r]
        dists[dists_idx_l] = x[dists_idx_r]
        pts3d[pts3d_idx_l] = x[pts3d_idx_r]
        return rvecs, tvecs, intrs, dists, pts3d

    def residuals(x):
        rvecs, tvecs, intrs, dists, pts3d = unpack(x)
        p2ds_proj = _project_per_obs(
            jnp.asarray(rvecs[obs_v]),
            jnp.asarray(tvecs[obs_v]),
            jnp.asarray(intrs[obs_v]),
            jnp.asarray(dists[obs_v]),
            jnp.asarray(pts3d[obs_n]),
        )
        return (np.asarray(p2ds_proj) - pts2d_obs).ravel()

    # Precompute the (row, col) coordinates of nonzero Jacobian entries. Each
    # observation contributes a dense (2, total_cols_per_obs) block whose
    # columns come from the rvec/tvec/intr/dist/pt3d slots it touches. Fixed
    # parameters are dropped via the `valid` mask.
    free_idx_map = np.full(values.size, -1, dtype=np.int64)
    free_idx_map[~fixed] = np.arange(n_free)
    cols_per_obs = np.concatenate(
        [
            rvecs_idx[obs_v],
            tvecs_idx[obs_v],
            intrs_idx[obs_v],
            dists_idx[obs_v],
            pts3d_idx[obs_n],
        ],
        axis=1,
    )
    free_cols_per_obs = free_idx_map[cols_per_obs]
    valid = free_cols_per_obs >= 0
    cols_nz = free_cols_per_obs[valid]
    obs_idx = np.broadcast_to(np.arange(n_obs)[:, None], cols_per_obs.shape)
    rows_x = 2 * obs_idx[valid]
    rows_y = rows_x + 1
    rows_combined = np.concatenate([rows_x, rows_y])
    cols_combined = np.concatenate([cols_nz, cols_nz])

    def jac(x):
        rvecs, tvecs, intrs, dists, pts3d = unpack(x)
        J_blocks = _jac_per_obs(
            jnp.asarray(rvecs[obs_v]),
            jnp.asarray(tvecs[obs_v]),
            jnp.asarray(intrs[obs_v]),
            jnp.asarray(dists[obs_v]),
            jnp.asarray(pts3d[obs_n]),
        )
        # tuple of 5 arrays, each (n_obs, 2, *param_dim); concat along last axis
        # to match `cols_per_obs` column ordering.
        J = np.concatenate([np.asarray(jb) for jb in J_blocks], axis=-1)
        data = np.concatenate([J[:, 0, :][valid], J[:, 1, :][valid]])
        return csr_matrix(
            (data, (rows_combined, cols_combined)),
            shape=(2 * n_obs, n_free),
        )

    res = least_squares(
        residuals,
        x,
        jac=jac,
        method="trf",
        tr_solver="lsmr",
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        **kwargs,
    )
    return res, unpack(res.x)


_project_per_obs_vmap = jax.vmap(project_one)


def _optx_residuals(x, args):
    """Pure-JAX residuals for Optimistix. Module-level so JAX caches the trace."""
    (
        values,
        free_idx,
        rvecs_idx,
        tvecs_idx,
        intrs_idx,
        dists_idx,
        pts3d_idx,
        obs_v,
        obs_n,
        pts2d_obs,
    ) = args
    full = values.at[free_idx].set(x)
    rvecs = full[rvecs_idx]
    tvecs = full[tvecs_idx]
    intrs = full[intrs_idx]
    dists = full[dists_idx]
    pts3d = full[pts3d_idx]
    p2d = _project_per_obs_vmap(
        rvecs[obs_v], tvecs[obs_v], intrs[obs_v], dists[obs_v], pts3d[obs_n]
    )
    return (p2d - pts2d_obs).ravel()


def bundle_adjust_optx(
    values: np.ndarray,
    fixed: np.ndarray,
    rvecs_idx: np.ndarray,
    tvecs_idx: np.ndarray,
    intrs_idx: np.ndarray,
    dists_idx: np.ndarray,
    pts3d_idx: np.ndarray,
    pts2d: np.ndarray,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    max_steps: int = 1000,
    linear_solver: str = "lsmr",
    lsmr_rtol: float = 1e-6,
    lsmr_atol: float = 1e-6,
    verbose: bool = False,
):
    """Bundle adjustment via Optimistix Levenberg--Marquardt.

    Same `prep_args` interface as `bundle_adjust`.

    `linear_solver`:
        - ``"lsmr"`` (default): matrix-free LSMR via JVPs (sparsity-friendly).
        - ``"normal_cg"``: matrix-free CG on the normal equations.
        - ``"qr"``: materialised dense QR (Optimistix default).
    """
    obs_v_np, obs_n_np = np.where(np.isfinite(pts2d).all(axis=-1))
    free_idx_np = np.where(~fixed)[0]

    args = (
        jnp.asarray(values),
        jnp.asarray(free_idx_np),
        jnp.asarray(rvecs_idx),
        jnp.asarray(tvecs_idx),
        jnp.asarray(intrs_idx),
        jnp.asarray(dists_idx),
        jnp.asarray(pts3d_idx),
        jnp.asarray(obs_v_np),
        jnp.asarray(obs_n_np),
        jnp.asarray(pts2d[obs_v_np, obs_n_np]),
    )

    if linear_solver == "lsmr":
        lin = lx.LSMR(rtol=lsmr_rtol, atol=lsmr_atol)
    elif linear_solver == "normal_cg":
        lin = lx.NormalCG(rtol=lsmr_rtol, atol=lsmr_atol)
    elif linear_solver == "qr":
        lin = lx.QR()
    else:
        raise ValueError(f"unknown linear_solver: {linear_solver}")

    solver = optx.LevenbergMarquardt(
        rtol=rtol,
        atol=atol,
        linear_solver=lin,
        verbose=verbose,
    )
    x0 = args[0][args[1]]
    sol = optx.least_squares(
        _optx_residuals,
        solver,
        x0,
        args=args,
        max_steps=max_steps,
        throw=False,
    )

    full_opt = args[0].at[args[1]].set(sol.value)
    out = tuple(
        np.asarray(full_opt[ix]) for ix in (args[2], args[3], args[4], args[5], args[6])
    )
    return sol, out
