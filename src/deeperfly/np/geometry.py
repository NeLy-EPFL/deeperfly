"""Multi-view geometry primitives (NumPy).

Conventions:

- Points carry their dimensionality in the last axis: ``pts3d`` has shape
  ``(..., 3)``; ``pts2d`` has shape ``(..., 2)``.
- Camera extrinsics are an axis-angle rotation vector ``rvec`` and a
  translation vector ``tvec``. A 3D world point ``X`` is mapped to camera
  coordinates as ``R(rvec) @ X + tvec``.
- Camera intrinsics ``intr`` are packed as ``[fx, ..., fy, cx, cy]`` (see
  :func:`intr_to_kmat`); distortion coefficients ``dists`` follow OpenCV's
  ordering ``[k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]``.
- A projection matrix ``pmat`` is the 3x4 product ``K @ [R | t]``.

Geometry functions take their primary operand (``pts3d``, ``pts2d``, ``rvec``,
``rmat``, ...) first and any camera parameters after, in the canonical order
``rvecs, tvecs, intrs, dists``.
"""

from __future__ import annotations

import cv2
import numpy as np
from jaxtyping import Float

# Below this squared rotation angle, sin/cos are evaluated via Taylor series
# to avoid catastrophic cancellation in ``1 - cos theta``.
_SMALL_THETA_SQ = 1e-8
# Below this value of ``sin theta``, the antisymmetric part of ``R`` is too
# small to recover the rotation axis and we fall back to the symmetric branch.
_NEAR_PI_SIN_THRESH = 1e-5


def intr_to_kmat(
    intr: Float[np.ndarray, "*batch 3"] | Float[np.ndarray, "*batch 4"],
) -> Float[np.ndarray, "*batch 3 3"]:
    """Build 3x3 camera intrinsic matrices from packed intrinsic parameters.

    Parameters
    ----------
    intr
        Packed intrinsic parameters of shape ``(..., 3)`` or ``(..., 4)``
        with the last dimensions corresponding to [fx, cx, cy] or
        [fx, fy, cx, cy], respectively.

    Returns
    -------
    Camera matrices ``K`` of shape ``(..., 3, 3)``.
    """
    kmat = np.zeros((*intr.shape[:-1], 3, 3))
    kmat[..., 0, 0] = intr[..., 0]
    kmat[..., 1, 1] = intr[..., -3]
    kmat[..., :2, 2] = intr[..., -2:]
    kmat[..., 2, 2] = 1.0
    return kmat


def rvec_to_rmat(
    rvec: Float[np.ndarray, "*batch 3"],
) -> Float[np.ndarray, "*batch 3 3"]:
    """Convert axis-angle rotation vectors to rotation matrices (Rodrigues).

    Implements ``R = I + a * W + b * W^2`` with ``W = skew(rvec)``,
    ``a = sin(theta) / theta`` and ``b = (1 - cos(theta)) / theta^2``. Working
    on the unnormalised axis avoids ``0/0`` at ``theta = 0``; evaluating ``a``
    and ``b`` from their Taylor expansions for small ``theta`` sidesteps the
    catastrophic cancellation in ``1 - cos(theta)``, keeping the result
    orthogonal to machine precision even for tiny rotations. ``W^2`` is
    expanded as ``rvec . rvec^T - theta^2 * I`` to avoid a batched matmul.

    Parameters
    ----------
    rvec
        Axis-angle rotation vectors of shape ``(..., 3)``. The direction is
        the rotation axis; the magnitude is the angle in radians.

    Returns
    -------
    Rotation matrices of shape ``(..., 3, 3)``.
    """
    rvec = np.asarray(rvec, dtype=float)
    theta_sq = np.einsum("...i,...i->...", rvec, rvec)
    theta = np.sqrt(theta_sq)
    small = theta_sq < _SMALL_THETA_SQ
    sinc_theta = np.where(  # sin(theta) / theta
        small,
        1 - theta_sq / 6 + theta_sq**2 / 120,
        np.sin(theta) / np.where(small, 1.0, theta),
    )
    cosc_theta = np.where(  # (1 - cos(theta)) / theta^2
        small,
        0.5 - theta_sq / 24 + theta_sq**2 / 720,
        (1 - np.cos(theta)) / np.where(small, 1.0, theta_sq),
    )
    skew = np.zeros((*rvec.shape[:-1], 3, 3))
    skew[..., 0, 1] = -rvec[..., 2]
    skew[..., 0, 2] = rvec[..., 1]
    skew[..., 1, 2] = -rvec[..., 0]
    skew[..., 1, 0] = rvec[..., 2]
    skew[..., 2, 0] = -rvec[..., 1]
    skew[..., 2, 1] = rvec[..., 0]
    outer = rvec[..., :, None] * rvec[..., None, :]
    # ``cos(theta) * I`` recovered from ``cosc_theta`` (no cancellation).
    diag = (1 - cosc_theta * theta_sq)[..., None, None] * np.eye(3)
    return (
        diag + sinc_theta[..., None, None] * skew + cosc_theta[..., None, None] * outer
    )


def rmat_to_rvec(
    rmat: Float[np.ndarray, "*batch 3 3"],
) -> Float[np.ndarray, "*batch 3"]:
    """Convert rotation matrices to axis-angle rotation vectors.

    Vectorised port of OpenCV's ``Rodrigues`` (matrix -> vector). The axis is
    read off the antisymmetric part ``R - R^T`` in the generic case, but that
    part vanishes at ``theta = pi`` (``R`` becomes symmetric), so near
    ``theta = pi`` the axis is instead recovered from the symmetric part
    ``(R + I) / 2`` with the signs disambiguated from the off-diagonal entries.

    Parameters
    ----------
    rmat
        Rotation matrices of shape ``(..., 3, 3)``.

    Returns
    -------
    Axis-angle rotation vectors of shape ``(..., 3)``.
    """
    r00, r01, r02 = rmat[..., 0, 0], rmat[..., 0, 1], rmat[..., 0, 2]
    r10, r11, r12 = rmat[..., 1, 0], rmat[..., 1, 1], rmat[..., 1, 2]
    r20, r21, r22 = rmat[..., 2, 0], rmat[..., 2, 1], rmat[..., 2, 2]

    rho = np.stack((r21 - r12, r02 - r20, r10 - r01), axis=-1)  # 2*sin(t)*axis
    sin_theta = np.linalg.norm(rho, axis=-1) / 2
    cos_theta = np.clip((r00 + r11 + r22 - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    near_pi = sin_theta < _NEAR_PI_SIN_THRESH

    # Generic case: rvec = (theta / (2 sin(theta))) * rho. Denominator guarded
    # where it would be undefined (handled by the near-pi branch below).
    rvec = rho * (theta / np.where(near_pi, 1.0, 2.0 * sin_theta))[..., None]

    # Near-pi: recover the axis from the diagonal of (R + I) / 2.
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
    axis_norm = np.linalg.norm(axis, axis=-1)
    rvec_pi = axis * (theta / np.where(axis_norm == 0, 1.0, axis_norm))[..., None]
    # cos > 0 implies theta ~ 0 (return zero vector); otherwise theta ~ pi.
    rvec_degenerate = np.where((cos_theta > 0)[..., None], 0.0, rvec_pi)

    return np.where(near_pi[..., None], rvec_degenerate, rvec)


def project_pmat(
    pts3d: Float[np.ndarray, "*pts 3"],
    pmats: Float[np.ndarray, "*cams 3 4"],
) -> Float[np.ndarray, "*cams *pts 2"]:
    """Project 3D world points to 2D image points using 3x4 projection matrices.

    Parameters
    ----------
    pts3d
        3D points of shape ``(*pts, 3)``.
    pmats
        Projection matrices of shape ``(*cams, 3, 4)`` -- typically
        ``K @ [R | t]``.

    Returns
    -------
    2D image points of shape ``(*cams, *pts, 2)``.
    """
    output_shape = (*pmats.shape[:-2], *pts3d.shape[:-1], 2)
    pmats_flat = pmats.reshape(-1, 3, 4)  # (C, 3, 4)
    pts_flat = pts3d.reshape(-1, 3).T  # (3, N)
    # (C, 3, 3) @ (3, N) + (C, 3, 1) -> (C, 3, N)
    pts2dh = pmats_flat[:, :, :3] @ pts_flat + pmats_flat[:, :, 3:]
    pts2d = (pts2dh[:, :2] / pts2dh[:, 2:]).transpose(0, 2, 1)  # (C, N, 2)
    return pts2d.reshape(output_shape)


def triangulate_dlt(
    pts2d: Float[np.ndarray, "V *pts 2"],
    pmats: Float[np.ndarray, "V 3 4"],
) -> Float[np.ndarray, "*pts 3"]:
    """Triangulate 3D points by direct linear transformation (DLT).

    For each point, stacks two rows per view of the linear system
    ``[x * p3 - p1; y * p3 - p2] @ [X, Y, Z, 1]^T = 0`` (where ``pi`` is the
    i-th row of ``pmat``) and solves for the homogeneous coordinates as the
    right-singular vector for the smallest singular value. The smallest
    right-singular vector of ``A`` equals the eigenvector of ``A^T A`` for the
    smallest eigenvalue, which is faster than SVD for the 4x4 problem. NaN
    observations are zeroed out; points with fewer than two valid views are
    returned as NaN.

    Parameters
    ----------
    pts2d
        2D observations of shape ``(V, *pts, 2)``. NaN entries indicate
        missing observations.
    pmats
        Projection matrices of shape ``(V, 3, 4)``.

    Returns
    -------
    Triangulated 3D points of shape ``(*pts, 3)``.
    """
    # (*pts, V, 2, 4)
    a = np.einsum("v...i,vj->...vij", pts2d, pmats[:, 2]) - pmats[:, :2]
    valid = np.moveaxis(np.isfinite(pts2d).all(axis=-1), 0, -1)  # (*pts, V)
    a[~valid] = 0
    a = a.reshape((*a.shape[:-3], -1, 4))  # (*pts, 2V, 4)
    # eigh on A^T A is faster than SVD of A and yields the same null vector.
    ata = np.einsum("...ij,...ik->...jk", a, a)  # (*pts, 4, 4)
    pts3dh = np.linalg.eigh(ata)[1][..., :, 0]  # (*pts, 4)
    pts3d = pts3dh[..., :3] / pts3dh[..., 3:]  # (*pts, 3)
    pts3d[valid.sum(axis=-1) < 2] = np.nan
    return pts3d


def distort(
    pts2d: Float[np.ndarray, "V *pts 2"],
    dists: Float[np.ndarray, "V K"],
) -> Float[np.ndarray, "V *pts 2"]:
    """Apply OpenCV-style radial + tangential + thin-prism distortion.

    For normalised image coordinates ``(x, y)`` with ``r^2 = x^2 + y^2``, the
    distortion model with up to 12 coefficients
    ``[k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]`` is

    .. code-block:: text

        x_d = x * (1 + k1 r^2 + k2 r^4 + k3 r^6)
                / (1 + k4 r^2 + k5 r^4 + k6 r^6)
              + 2 p1 x y + p2 (r^2 + 2 x^2) + s1 r^2 + s2 r^4
        y_d = y * (...) / (...)
              + p1 (r^2 + 2 y^2) + 2 p2 x y + s3 r^2 + s4 r^4

    Coefficients beyond ``K`` are taken as zero, matching ``cv2.projectPoints``.

    Parameters
    ----------
    pts2d
        Normalised 2D coordinates of shape ``(V, *pts, 2)``.
    dists
        Distortion coefficients of shape ``(V, K)`` with ``K`` in
        ``{0, 1, ..., 12}``.

    Returns
    -------
    Distorted 2D coordinates of shape ``(V, *pts, 2)``.
    """
    n = dists.shape[-1]
    if n == 0:
        return pts2d

    dists = dists.T  # (K, V)
    pts2d_t = pts2d.T  # (2, *reversed pts, V)
    x, y = pts2d_t[0], pts2d_t[1]
    x2, y2 = x * x, y * y
    r2 = x2 + y2

    # Radial polynomial (even powers of r), as numerator / denominator.
    num = 1.0 + dists[0] * r2
    r4 = r6 = None
    if n >= 2:
        r4 = r2 * r2
        num = num + dists[1] * r4
    if n >= 5:
        r6 = r4 * r2
        num = num + dists[4] * r6
    den = 1.0
    if n >= 6:
        den = den + dists[5] * r2
    if n >= 7:
        den = den + dists[6] * r4
    if n >= 8:
        den = den + dists[7] * r6
    mult = num / den

    # Tangential and thin-prism additive terms.
    add_x = np.zeros_like(x)
    add_y = np.zeros_like(y)
    if n >= 3:
        xy = x * y
        add_x = 2 * dists[2] * xy
        add_y = dists[2] * (r2 + 2 * y2)
    if n >= 4:
        add_x = add_x + dists[3] * (r2 + 2 * x2)
        add_y = add_y + 2 * dists[3] * xy
    if n >= 9:
        add_x = add_x + dists[8] * r2
    if n >= 10:
        add_x = add_x + dists[9] * r4
    if n >= 11:
        add_y = add_y + dists[10] * r2
    if n >= 12:
        add_y = add_y + dists[11] * r4

    return np.stack([x * mult + add_x, y * mult + add_y], axis=0).T


def project_full(
    pts3d: Float[np.ndarray, "*pts 3"],
    rvecs: Float[np.ndarray, "V 3"],
    tvecs: Float[np.ndarray, "V 3"],
    intrs: Float[np.ndarray, "V P"] | Float[np.ndarray, "P"],
    dists: Float[np.ndarray, "V K"] | Float[np.ndarray, "K"],
) -> Float[np.ndarray, "V *pts 2"]:
    """Project 3D world points to 2D image points through full camera models.

    Composes the pinhole projection ``X_cam = R(rvec) X + tvec``, perspective
    division ``xy = X_cam[:2] / X_cam[2]``, distortion via :func:`distort`, and
    the affine intrinsics ``x_pix = fx * x + cx`` (and analogously for ``y``).

    Parameters
    ----------
    pts3d
        3D world points of shape ``(*pts, 3)``.
    rvecs
        Axis-angle rotation vectors of shape ``(V, 3)``.
    tvecs
        Translation vectors of shape ``(V, 3)``.
    intrs
        Packed intrinsics of shape ``(V, P)`` or ``(P,)`` (shared); see
        :func:`intr_to_kmat`.
    dists
        Distortion coefficients of shape ``(V, K)`` or ``(K,)`` (shared).

    Returns
    -------
    Projected 2D image points of shape ``(V, *pts, 2)``.
    """
    pts_dims = pts3d.shape[:-1]
    rmats = rvec_to_rmat(rvecs)
    xyz = ((rmats @ pts3d.T) + tvecs[..., None]).transpose(0, 2, 1)
    xy = xyz[:, :, :2] / xyz[:, :, 2:3]
    xy = distort(xy, dists)
    intr_shape = (*intrs.shape[:-1], *(1,) * len(pts_dims), 2)
    return xy * intrs[..., [0, -3]].reshape(intr_shape) + intrs[..., -2:].reshape(
        intr_shape
    )


def project_full_cv2(
    pts3d: Float[np.ndarray, "*pts 3"],
    rvecs: Float[np.ndarray, "V 3"] | Float[np.ndarray, "3"],
    tvecs: Float[np.ndarray, "V 3"] | Float[np.ndarray, "3"],
    intrs: Float[np.ndarray, "V P"] | Float[np.ndarray, "P"],
    dists: Float[np.ndarray, "V K"] | Float[np.ndarray, "K"],
) -> tuple[Float[np.ndarray, "V *pts 2"], np.ndarray]:
    """Reference projection using :func:`cv2.projectPoints` (per view).

    Single-view inputs (all 1D) are passed straight through; multi-view inputs
    are broadcast against the number of views and iterated. Useful for
    validating :func:`project_full` against OpenCV.

    Parameters
    ----------
    pts3d
        3D world points of shape ``(*pts, 3)``.
    rvecs, tvecs, intrs, dists
        Per-view camera parameters; either 1D (single shared camera) or 2D
        with a leading view axis.

    Returns
    -------
    Tuple ``(pts2d, jac)`` where ``pts2d`` has shape ``(V, *pts, 2)`` and
    ``jac`` is the stacked OpenCV Jacobian.
    """
    pts_dims = pts3d.shape[:-1]
    pts3d = pts3d.reshape((-1, 3))
    kmats = intr_to_kmat(intrs)

    if rvecs.ndim == tvecs.ndim == intrs.ndim == dists.ndim == 1:
        pts2d, jac = cv2.projectPoints(pts3d, rvecs, tvecs, kmats, dists)
        return pts2d.reshape((*pts_dims, 2)), jac

    assert max(rvecs.ndim, tvecs.ndim, intrs.ndim, dists.ndim) == 2
    n_views = max(
        *rvecs.shape[:-1], *tvecs.shape[:-1], *intrs.shape[:-1], *dists.shape[:-1]
    )
    per_view = zip(
        rvecs if rvecs.ndim > 1 else (rvecs,) * n_views,
        tvecs if tvecs.ndim > 1 else (tvecs,) * n_views,
        kmats if kmats.ndim > 2 else (kmats,) * n_views,
        dists if dists.ndim > 1 else (dists,) * n_views,
        strict=True,
    )
    pts2d, jac = zip(*(cv2.projectPoints(pts3d, *args) for args in per_view))
    pts2d = np.array(pts2d).reshape((n_views, *pts_dims, 2))
    jac = np.array(jac)
    return pts2d, jac
