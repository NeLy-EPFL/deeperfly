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
    output_shape = pmats.shape[:-2] + pts3d.shape[:-1] + (2,)
    pts3dh = np.concatenate((pts3d, np.ones((*pts3d.shape[:-1], 1))), axis=-1)
    pts2dh = np.einsum("ijk,...k->i...j", pmats.reshape((-1, 3, 4)), pts3dh)
    pts2d = pts2dh[..., :2] / pts2dh[..., 2:]
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
    amat = (
        np.einsum("v...i,vj->...vij", pts2d, pmats[:, 2]) - pmats[:, :2]
    )  # (*dims, views, 2, 4)
    # Zero out rows from NaN views — zero rows are ignored by SVD least-squares
    valid = np.moveaxis(np.isfinite(pts2d).all(axis=-1), 0, -1)  # (*dims, views)
    amat[~valid] = 0
    amat = amat.reshape((*amat.shape[:-3], -1, 4))  # (*dims, views*2, 4)
    pts3dh = np.linalg.svd(amat)[-1][..., -1, :]  # (*dims, 4)
    result = pts3dh[..., :3] / pts3dh[..., 3:]  # (*dims, 3)
    result[valid.sum(axis=-1) < 2] = np.nan
    return result
