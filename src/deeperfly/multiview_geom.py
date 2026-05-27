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
