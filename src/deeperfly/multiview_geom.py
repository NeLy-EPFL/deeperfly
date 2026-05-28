import cv2
import numpy as np


def project(pmats: np.ndarray, pts3d: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D using a projection matrix.

    Parameters
    ----------
    pmats : np.ndarray
        Projection matrices of shape (*dims1, 3, 4).
    pts3d : np.ndarray
        3D points of shape (*dims2, 3).
    Returns
    -------
    np.ndarray
        2D projected points of shape (*dims1, *dims2, 2).
    """
    output_shape = (*pmats.shape[:-2], *pts3d.shape[:-1], 2)
    pmats_flat = pmats.reshape(-1, 3, 4)  # (C, 3, 4)
    pts_flat = pts3d.reshape(-1, 3).T  # (3, N)
    # (C, 3, 3) @ (3, N) + (C, 3, 1)  →  (C, 3, N)
    pts2dh = pmats_flat[:, :, :3] @ pts_flat + pmats_flat[:, :, 3:]
    pts2d = (pts2dh[:, :2] / pts2dh[:, 2:]).transpose(0, 2, 1)  # (C, N, 2)
    return pts2d.reshape(output_shape)


def triangulate_dlt(pmats: np.ndarray, pts2d: np.ndarray) -> np.ndarray:
    """Triangulate 3D points from 2D correspondences using DLT.

    Parameters
    ----------
    pmats : np.ndarray
        Projection matrices of shape (views, 3, 4).
    pts2d : np.ndarray
        2D points of shape (views, *dims, 2).
    Returns
    -------
    np.ndarray
        Triangulated 3D points of shape (*dims, 3).
    """
    a = (
        np.einsum("v...i,vj->...vij", pts2d, pmats[:, 2]) - pmats[:, :2]
    )  # (*dims, views, 2, 4)
    # zero out rows containing nans
    valid = np.moveaxis(np.isfinite(pts2d).all(axis=-1), 0, -1)  # (*dims, views)
    a[~valid] = 0
    a = a.reshape((*a.shape[:-3], -1, 4))  # (*dims, views*2, 4)
    # eigh on AᵀA (4×4 symmetric PSD) is faster than SVD of A and yields the
    # same right-singular vector for the smallest singular value
    ata = np.einsum("...ij,...ik->...jk", a, a)  # (*dims, 4, 4)
    pts3dh = np.linalg.eigh(ata)[1][..., :, 0]  # (*dims, 4)
    result = pts3dh[..., :3] / pts3dh[..., 3:]  # (*dims, 3)
    result[valid.sum(axis=-1) < 2] = np.nan
    return result


def rvec2mat(rvec: np.ndarray):
    """Convert rotation vectors to rotation matrices (Rodrigues' formula).

    Written on the *unnormalized* axis as ``R = I + a·W + b·W²`` where
    ``W = skew(rvec)``, ``a = sinθ/θ`` and ``b = (1 - cosθ)/θ²``. This avoids
    normalizing the axis (no ``0/0`` at ``θ = 0``) and, by evaluating ``a`` and
    ``b`` from their Taylor series for small ``θ``, sidesteps the catastrophic
    cancellation in ``1 - cosθ`` — keeping the result orthogonal down to
    machine precision even for tiny rotations. ``W²`` is expanded as the outer
    product ``rvec·rvecᵀ - θ²·I`` to avoid a batched matrix product.

    Parameters
    ----------
    rvec : np.ndarray
        Rotation vectors of shape (..., 3).
    Returns
    -------
    np.ndarray
        Rotation matrices of shape (..., 3, 3).
    """
    rvec = np.asarray(rvec, dtype=float)
    theta2 = np.einsum("...i,...i->...", rvec, rvec)  # θ²  (...)
    theta = np.sqrt(theta2)
    small = theta2 < 1e-8  # θ < 1e-4: use Taylor series instead of sin/cos
    a = np.where(  # sinθ/θ
        small,
        1 - theta2 / 6 + theta2**2 / 120,
        np.sin(theta) / np.where(small, 1.0, theta),
    )
    b = np.where(  # (1 - cosθ)/θ²
        small,
        0.5 - theta2 / 24 + theta2**2 / 720,
        (1 - np.cos(theta)) / np.where(small, 1.0, theta2),
    )
    W = np.zeros((*rvec.shape[:-1], 3, 3))  # skew(rvec), (..., 3, 3)
    W[..., 0, 1] = -rvec[..., 2]
    W[..., 0, 2] = rvec[..., 1]
    W[..., 1, 2] = -rvec[..., 0]
    W[..., 1, 0] = rvec[..., 2]
    W[..., 2, 0] = -rvec[..., 1]
    W[..., 2, 1] = rvec[..., 0]
    vvt = rvec[..., :, None] * rvec[..., None, :]  # rvec·rvecᵀ, (..., 3, 3)
    diag = (1 - b * theta2)[..., None, None] * np.eye(
        3
    )  # cosθ·I, via b (no cancellation)
    return diag + a[..., None, None] * W + b[..., None, None] * vvt


def rmat2vec(rmat: np.ndarray):
    """Convert rotation matrices to rotation vectors.

    Vectorized port of OpenCV's ``Rodrigues`` (matrix -> vector). The axis is
    read off the antisymmetric part ``R - Rᵀ`` in the generic case, but that part
    vanishes at a 180° rotation (``R`` becomes symmetric), so near ``θ = π`` the
    axis is instead recovered from the symmetric part ``(R + I) / 2`` with the
    signs disambiguated from the off-diagonal entries.

    Parameters
    ----------
    rmat : np.ndarray
        Rotation matrices of shape (..., 3, 3).
    Returns
    -------
    np.ndarray
        Rotation vectors of shape (..., 3).
    """
    r00, r01, r02 = rmat[..., 0, 0], rmat[..., 0, 1], rmat[..., 0, 2]
    r10, r11, r12 = rmat[..., 1, 0], rmat[..., 1, 1], rmat[..., 1, 2]
    r20, r21, r22 = rmat[..., 2, 0], rmat[..., 2, 1], rmat[..., 2, 2]

    rho = np.stack((r21 - r12, r02 - r20, r10 - r01), axis=-1)  # 2·sinθ·axis
    s = np.linalg.norm(rho, axis=-1) / 2  # sinθ ≥ 0
    c = np.clip((r00 + r11 + r22 - 1) / 2, -1.0, 1.0)  # cosθ
    theta = np.arccos(c)  # θ ∈ [0, π]
    near_pi = s < 1e-5  # antisymmetric part too small to give the axis

    # Generic case: rvec = θ/(2·sinθ) · rho  (denominator guarded where unused).
    rvec = rho * (theta / np.where(near_pi, 1.0, 2.0 * s))[..., None]

    # Degenerate case: θ ≈ 0 (→ 0) or θ ≈ π (axis from the symmetric part).
    ax = np.sqrt(np.maximum((r00 + 1.0) / 2, 0.0))
    ay = np.sqrt(np.maximum((r11 + 1.0) / 2, 0.0)) * np.where(r01 < 0, -1.0, 1.0)
    az = np.sqrt(np.maximum((r22 + 1.0) / 2, 0.0)) * np.where(r02 < 0, -1.0, 1.0)
    flip = (
        (np.abs(ax) < np.abs(ay))
        & (np.abs(ax) < np.abs(az))
        & ((r12 > 0) != (ay * az > 0))
    )
    az = np.where(flip, -az, az)
    axis = np.stack((ax, ay, az), axis=-1)
    norm = np.linalg.norm(axis, axis=-1)
    rvec_pi = axis * (theta / np.where(norm == 0, 1.0, norm))[..., None]
    rvec_deg = np.where((c > 0)[..., None], 0.0, rvec_pi)  # c > 0 ⇒ θ ≈ 0 ⇒ zero

    return np.where(near_pi[..., None], rvec_deg, rvec)


def distort(pts2d: np.ndarray, dists: np.ndarray):
    """Distort 2D points

    Parameters
    ----------
    pts2d : np.ndarray
        2D points of shape (views, ..., 2).
    dists : np.ndarray
        Distortion coefficients of shape (views, {0...12}).

    Returns
    -------
    np.ndarray
        Distorted 2D points of shape (views, ..., 2).

    """
    n_params = dists.shape[-1]

    if n_params == 0:
        return pts2d

    # k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4
    dists = dists.T  # (K, views)
    pts2d = pts2d.T  # (2, ..., views)
    x = pts2d[0]  # (..., views)
    y = pts2d[1]  # (..., views)
    x2 = x * x  # (..., views)
    y2 = y * y  # (..., views)
    r2 = x2 + y2  # (..., views)
    r2 = x2 + y2  # (..., views)

    # k1
    mult = 1.0 + dists[0] * r2
    add = None

    if n_params >= 2:
        r4 = r2 * r2
        # k2
        mult = mult + dists[1] * r4
        if n_params >= 3:
            xy = x * y
            # p1
            add = (2 * dists[2] * xy, dists[2] * (r2 + 2 * y2))
            if n_params >= 4:
                # p2
                add = (add[0] + dists[3] * (r2 + 2 * x2), add[1] + 2 * dists[3] * xy)
                if n_params >= 5:
                    r6 = r4 * r2
                    # k3
                    mult = mult + dists[4] * r6
                    if n_params >= 6:
                        # k4
                        den = 1.0 + dists[5] * r2
                        if n_params >= 7:
                            # k5
                            den = den + dists[6] * r4
                            if n_params >= 8:
                                # k6
                                den = den + dists[7] * r6
                                if n_params >= 9:
                                    # s1
                                    add = (add[0] + dists[8] * r2, add[1])
                                    if n_params >= 10:
                                        # s2
                                        add = (add[0] + dists[9] * r4, add[1])
                                        if n_params >= 11:
                                            # s3
                                            add = (add[0], add[1] + dists[10] * r2)
                                            if n_params >= 12:
                                                # s4
                                                add = (add[0], add[1] + dists[11] * r4)
                        mult = mult / den

    pts2d_dist = pts2d * mult

    if add is not None:
        pts2d_dist = pts2d_dist + add

    return pts2d_dist.T


def intr2kmat(intr: np.ndarray):
    """Convert intrinsic parameters to camera matrix.

    Parameters
    ----------
    intr : np.ndarray
        Intrinsic parameters of shape (..., 4) or (..., 5).

    Returns
    -------
    np.ndarray
        Camera matrix of shape (..., 3, 3).
    """
    kmat = np.zeros((*intr.shape[:-1], 3, 3))
    kmat[..., 0, 0] = intr[..., 0]
    kmat[..., 1, 1] = intr[..., -3]
    kmat[..., :2, 2] = intr[..., -2:]
    kmat[..., 2, 2] = 1.0
    return kmat


def project_full_cv2(
    pts3d: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    intrs: np.ndarray,
    dists: np.ndarray,
):
    dims = pts3d.shape[:-1]
    pts3d = pts3d.reshape((-1, 3))
    kmats = intr2kmat(intrs)

    if rvecs.ndim == tvecs.ndim == intrs.ndim == dists.ndim == 1:
        pts2d, jac = cv2.projectPoints(pts3d, rvecs, tvecs, kmats, dists)
        return pts2d.reshape((*dims, 2)), jac

    assert max(rvecs.ndim, tvecs.ndim, intrs.ndim, dists.ndim) == 2
    n_views = max(
        *rvecs.shape[:-1], *tvecs.shape[:-1], *intrs.shape[:-1], *dists.shape[:-1]
    )
    it = zip(
        rvecs if rvecs.ndim > 1 else (rvecs,) * n_views,
        tvecs if tvecs.ndim > 1 else (tvecs,) * n_views,
        kmats if kmats.ndim > 2 else (kmats,) * n_views,
        dists if dists.ndim > 1 else (dists,) * n_views,
        strict=True,
    )
    pts2d, jac = zip(*(cv2.projectPoints(pts3d, *args) for args in it))
    pts2d = np.array(pts2d).reshape((n_views, *dims, 2))
    jac = np.array(jac)
    return pts2d, jac


def project_full(
    pts3d: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    intrs: np.ndarray,
    dists: np.ndarray,
):
    dims = pts3d.shape[:-1]
    rmats = rvec2mat(rvecs)
    xyz = ((rmats @ pts3d.T) + tvecs[..., None]).transpose(0, 2, 1)
    xy = xyz[:, :, :2] / xyz[:, :, 2:3]
    xy = distort(xy, dists)
    shape = (*intrs.shape[:-1], *(1,) * len(dims), 2)
    xy = xy * intrs[..., [0, -3]].reshape(shape) + intrs[..., -2:].reshape(shape)
    return xy
