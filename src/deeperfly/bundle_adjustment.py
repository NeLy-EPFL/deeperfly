import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

from .multiview_geom import rvecs2rmats


def bundle_adjustment(
    pts2d: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    intrinsics: np.ndarray,
    pts3d: np.ndarray,
    distortions: np.ndarray | None = None,
    rvec_idx: np.ndarray | None = None,
    tvec_idx: np.ndarray | None = None,
    intr_idx: np.ndarray | None = None,
    dist_idx: np.ndarray | None = None,
    fix_rvecs: np.ndarray | bool | None = None,
    fix_tvecs: np.ndarray | bool | None = None,
    fix_intrinsics: np.ndarray | bool | None = None,
    fix_distortions: np.ndarray | bool | None = None,
    fix_pts3d: np.ndarray | bool | None = None,
    **lsq_kwargs,
) -> dict:
    """Minimize reprojection error over camera poses, intrinsics and 3D points.

    Parameters
    ----------
    pts2d : np.ndarray, shape (V, N, 2)
        Observed 2D points; NaN entries are treated as missing.
    rvecs : np.ndarray, shape (R, 3)
        Initial Rodrigues rotation vectors (world -> camera) per rotation group.
    tvecs : np.ndarray, shape (T, 3)
        Initial translation vectors per translation group.
    intrinsics : np.ndarray, shape (I, 3) or (I, 4)
        Initial intrinsics per group. `(I, 3) = [f, cx, cy]` enforces fx=fy;
        `(I, 4) = [fx, fy, cx, cy]` allows them to differ.
    pts3d : np.ndarray, shape (N, 3)
        Initial 3D point positions.
    distortions : np.ndarray, shape (D, K), optional
        Brown-Conrady coefficients per group in OpenCV order
        `[k1, k2, p1, p2, k3, k4, k5, k6]`. Use `K in {0, 1, 2, 4, 5, 8}`.
        For per-group variable K, pad with zeros and fix those entries.
    rvec_idx, tvec_idx, intr_idx, dist_idx : np.ndarray, shape (V,), optional
        View -> group mapping for each parameter family. Default: `arange(V)`
        if the group count equals V, zeros if the group count is 1, otherwise
        must be supplied explicitly.
    fix_rvecs, fix_tvecs, fix_intrinsics, fix_distortions, fix_pts3d : bool or
            np.ndarray of bool, optional
        Boolean mask of entries to hold fixed (`True` = fixed). Any shape
        broadcastable to the parameter array is accepted (e.g. `True` to fix
        the whole array, or shape `(1, 4)` to fix the same columns for every
        intrinsics group). `None` (default) means every entry is free.
    **lsq_kwargs
        Forwarded to :func:`scipy.optimize.least_squares` (e.g. `loss`,
        `ftol`, `verbose`, `max_nfev`).

    Returns
    -------
    dict
        Optimized `rvecs`, `tvecs`, `intrinsics`, `distortions`, `pts3d`,
        plus `result` (the `scipy.optimize.OptimizeResult`).
    """
    pts2d = np.asarray(pts2d, dtype=float)
    V, _, _ = pts2d.shape

    init = {
        "rvecs": np.array(rvecs, dtype=float),
        "tvecs": np.array(tvecs, dtype=float),
        "intrinsics": np.array(intrinsics, dtype=float),
        "pts3d": np.array(pts3d, dtype=float),
        "distortions": np.array(
            distortions if distortions is not None else np.zeros((0, 0)),
            dtype=float,
        ),
    }
    has_dist = init["distortions"].size > 0

    rvec_idx = _resolve_idx(rvec_idx, V, init["rvecs"].shape[0], "rvec_idx")
    tvec_idx = _resolve_idx(tvec_idx, V, init["tvecs"].shape[0], "tvec_idx")
    intr_idx = _resolve_idx(intr_idx, V, init["intrinsics"].shape[0], "intr_idx")
    dist_idx = (
        _resolve_idx(dist_idx, V, init["distortions"].shape[0], "dist_idx")
        if has_dist
        else None
    )

    obs_v, obs_n = np.where(np.isfinite(pts2d).all(axis=-1))
    pts2d_obs = pts2d[obs_v, obs_n]

    free_masks = {
        "rvecs": _free_mask(fix_rvecs, init["rvecs"].shape),
        "tvecs": _free_mask(fix_tvecs, init["tvecs"].shape),
        "intrinsics": _free_mask(fix_intrinsics, init["intrinsics"].shape),
        "distortions": _free_mask(fix_distortions, init["distortions"].shape),
        "pts3d": _free_mask(fix_pts3d, init["pts3d"].shape),
    }
    packer = _ParamPacker(free_masks)
    x0 = packer.pack(init)

    def residuals(x):
        st = packer.unpack(x, init)
        proj = _project(
            st["rvecs"],
            st["tvecs"],
            st["intrinsics"],
            st["distortions"] if has_dist else None,
            st["pts3d"],
            rvec_idx,
            tvec_idx,
            intr_idx,
            dist_idx,
            obs_v,
            obs_n,
        )
        return (proj - pts2d_obs).ravel()

    lsq_kwargs.setdefault("method", "trf")
    lsq_kwargs.setdefault("x_scale", "jac")
    lsq_kwargs.setdefault(
        "jac_sparsity",
        _build_jac_sparsity(
            obs_v, obs_n, rvec_idx, tvec_idx, intr_idx, dist_idx, packer, has_dist
        ),
    )

    result = least_squares(residuals, x0, **lsq_kwargs)
    out = packer.unpack(result.x, init)
    out["result"] = result
    return out


def _resolve_idx(idx, V, n_groups, name):
    if idx is not None:
        idx = np.asarray(idx, dtype=int)
        if idx.shape != (V,):
            raise ValueError(f"{name} must have shape ({V},), got {idx.shape}")
        return idx
    if n_groups == 1:
        return np.zeros(V, dtype=int)
    if n_groups == V:
        return np.arange(V)
    raise ValueError(
        f"Cannot infer {name}: V={V} but group count={n_groups}; pass it explicitly."
    )


def _free_mask(fix, shape):
    if int(np.prod(shape)) == 0:
        return np.zeros(shape, dtype=bool)
    if fix is None:
        return np.ones(shape, dtype=bool)
    if np.isscalar(fix):
        return np.full(shape, not bool(fix))
    return np.ascontiguousarray(~np.broadcast_to(np.asarray(fix, dtype=bool), shape))


def _project(
    rvecs,
    tvecs,
    intrinsics,
    distortions,
    pts3d,
    rvec_idx,
    tvec_idx,
    intr_idx,
    dist_idx,
    obs_v,
    obs_n,
):
    """Project the (view, point) pairs given by `obs_v`, `obs_n`.

    Returns array of shape `(n_obs, 2)`.
    """
    rmats = rvecs2rmats(rvecs)  # (R, 3, 3)
    Xc = (
        np.einsum("oij,oj->oi", rmats[rvec_idx[obs_v]], pts3d[obs_n])
        + tvecs[tvec_idx[obs_v]]
    )  # (n_obs, 3)
    xn = Xc[:, 0] / Xc[:, 2]
    yn = Xc[:, 1] / Xc[:, 2]

    if distortions is not None:
        xn, yn = _distort(xn, yn, distortions[dist_idx[obs_v]])

    intr = intrinsics[intr_idx[obs_v]]  # (n_obs, 3 or 4)
    if intr.shape[1] == 3:
        f, cx, cy = intr[:, 0], intr[:, 1], intr[:, 2]
        u = f * xn + cx
        v = f * yn + cy
    else:
        fx, fy, cx, cy = intr[:, 0], intr[:, 1], intr[:, 2], intr[:, 3]
        u = fx * xn + cx
        v = fy * yn + cy
    return np.stack([u, v], axis=-1)


def _distort(xn, yn, k):
    """Brown-Conrady distortion. k has shape (..., K), K in {1, 2, 4, 5, 8}."""
    K = k.shape[-1]
    r2 = xn * xn + yn * yn

    num = 1.0 + k[..., 0] * r2
    if K >= 2:
        num = num + k[..., 1] * r2 * r2
    if K >= 5:
        num = num + k[..., 4] * r2 * r2 * r2
    den = 1.0
    if K >= 6:
        den = 1.0 + k[..., 5] * r2
    if K >= 7:
        den = den + k[..., 6] * r2 * r2
    if K >= 8:
        den = den + k[..., 7] * r2 * r2 * r2
    radial = num / den

    xd = xn * radial
    yd = yn * radial
    if K >= 4:
        p1, p2 = k[..., 2], k[..., 3]
        xy = xn * yn
        xd = xd + 2 * p1 * xy + p2 * (r2 + 2 * xn * xn)
        yd = yd + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xy
    return xd, yd


class _ParamPacker:
    """Pack/unpack only the free (non-fixed) entries of a set of named arrays."""

    def __init__(self, free_masks):
        self.free_masks = free_masks
        self.shapes = {n: m.shape for n, m in free_masks.items()}
        self.slices = {}
        offset = 0
        for name, mask in free_masks.items():
            n_free = int(mask.sum())
            self.slices[name] = slice(offset, offset + n_free)
            offset += n_free
        self.n_free = offset

    def pack(self, full):
        x = np.empty(self.n_free)
        for name, arr in full.items():
            x[self.slices[name]] = arr[self.free_masks[name]]
        return x

    def unpack(self, x, init):
        out = {name: arr.copy() for name, arr in init.items()}
        for name, sl in self.slices.items():
            out[name][self.free_masks[name]] = x[sl]
        return out


def _build_jac_sparsity(
    obs_v, obs_n, rvec_idx, tvec_idx, intr_idx, dist_idx, packer, has_dist
):
    n_obs = len(obs_v)

    free_idx = {}
    for name, sl in packer.slices.items():
        idx = np.full(packer.shapes[name], -1, dtype=int)
        mask = packer.free_masks[name]
        idx[mask] = np.arange(sl.start, sl.stop)
        free_idx[name] = idx

    blocks = [
        free_idx["rvecs"][rvec_idx[obs_v]],
        free_idx["tvecs"][tvec_idx[obs_v]],
        free_idx["intrinsics"][intr_idx[obs_v]],
        free_idx["pts3d"][obs_n],
    ]
    if has_dist:
        blocks.append(free_idx["distortions"][dist_idx[obs_v]])
    cols = np.concatenate(blocks, axis=1)  # (n_obs, n_cols_per_obs)
    n_per = cols.shape[1]

    # Each obs contributes 2 residual rows depending on the same columns.
    rows = np.repeat(np.arange(2 * n_obs), n_per)
    cols_flat = np.repeat(cols, 2, axis=0).ravel()

    keep = cols_flat >= 0
    return csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.int8), (rows[keep], cols_flat[keep])),
        shape=(2 * n_obs, packer.n_free),
    )
