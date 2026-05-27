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


def rvecs2rmats(rvecs: np.ndarray):
    """Convert rotation vectors to rotation matrices.

    Parameters
    ----------
    rvecs : np.ndarray
        Rotation vectors of shape (..., 3).
    Returns
    -------
    np.ndarray
        Rotation matrices of shape (..., 3, 3).
    """
    theta = np.linalg.norm(rvecs, axis=-1, keepdims=True)  # (..., 1)
    k = rvecs / np.where(theta > 0, theta, 1.0)  # (..., 3)
    K = np.zeros((*rvecs.shape[:-1], 3, 3))  # (..., 3, 3)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] = k[..., 1]
    K[..., 1, 2] = -k[..., 0]
    K[..., 1, 0] = k[..., 2]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] = k[..., 0]
    return (
        np.sin(theta[..., None]) * K
        + (1 - np.cos(theta[..., None])) * (K @ K)
        + np.eye(3)
    )


def rmats2rvecs(rmats: np.ndarray):
    """Convert rotation matrices to rotation vectors.

    Parameters
    ----------
    rmats : np.ndarray
        Rotation matrices of shape (..., 3, 3).
    Returns
    -------
    np.ndarray
        Rotation vectors of shape (..., 3).
    """
    rho = np.stack(
        (
            rmats[..., 2, 1] - rmats[..., 1, 2],
            rmats[..., 0, 2] - rmats[..., 2, 0],
            rmats[..., 1, 0] - rmats[..., 0, 1],
        ),
        axis=-1,
    )  # (..., 3)
    s = np.linalg.norm(rho, axis=-1) / 2  # (...,)
    c = (np.trace(rmats, axis1=-2, axis2=-1) - 1) / 2  # (...,)
    theta = np.arctan2(s, c)  # (...,)
    return rho * (0.5 / np.sinc(theta / np.pi))[..., None]


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
