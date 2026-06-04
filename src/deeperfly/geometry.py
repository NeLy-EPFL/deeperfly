"""Multi-view geometry primitives (JAX, pinned to the CPU).

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
``rvecs, tvecs, intrs, dists``. All functions are JIT- and grad-friendly.

The batched public functions are :func:`cpu_jit`-wrapped (see
:mod:`deeperfly._jax_cpu`); deeperfly installs only CPU JAX, so this camera-algebra
runs on the CPU -- the tiny arrays don't benefit from a GPU. The ``*_one`` variants
operate on a single observation (no leading batch axes) and are designed to be
composed with
:func:`jax.vmap` and :func:`jax.jacfwd` for bundle adjustment; the batched
public functions are themselves thin :func:`jax.vmap` wrappers around them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._jax_cpu import cpu_jit

# Below this squared rotation angle, sin/cos are evaluated via Taylor series
# to avoid catastrophic cancellation in ``1 - cos theta``.
_SMALL_THETA_SQ = 1e-8
# Below this value of ``sin theta``, the antisymmetric part of ``R`` is too
# small to recover the rotation axis and we fall back to the symmetric branch.
_NEAR_PI_SIN_THRESH = 1e-5


# -- per-observation primitives ---------------------------------------------


def rvec_to_rmat_one(rvec: Float[Array, "3"]) -> Float[Array, "3 3"]:
    """Rodrigues' rotation for a single rotation vector.

    Single-instance variant of :func:`rvec_to_rmat` for use under
    :func:`jax.vmap` and :func:`jax.jacfwd`.
    """
    theta_sq = jnp.dot(rvec, rvec)
    theta = jnp.sqrt(theta_sq)
    small = theta_sq < _SMALL_THETA_SQ
    sinc_theta = jnp.where(
        small,
        1 - theta_sq / 6 + theta_sq**2 / 120,
        jnp.sin(theta) / jnp.where(small, 1.0, theta),
    )
    cosc_theta = jnp.where(
        small,
        0.5 - theta_sq / 24 + theta_sq**2 / 720,
        (1 - jnp.cos(theta)) / jnp.where(small, 1.0, theta_sq),
    )
    rx, ry, rz = rvec[0], rvec[1], rvec[2]
    skew = jnp.array(
        [
            [0.0, -rz, ry],
            [rz, 0.0, -rx],
            [-ry, rx, 0.0],
        ]
    )
    outer = jnp.outer(rvec, rvec)
    return (
        (1 - cosc_theta * theta_sq) * jnp.eye(3)
        + sinc_theta * skew
        + cosc_theta * outer
    )


def rmat_to_rvec_one(rmat: Float[Array, "3 3"]) -> Float[Array, "3"]:
    """Rotation matrix to axis-angle vector for a single rotation.

    Single-instance variant of :func:`rmat_to_rvec`.
    """
    r00, r01, r02 = rmat[0, 0], rmat[0, 1], rmat[0, 2]
    r10, r11, r12 = rmat[1, 0], rmat[1, 1], rmat[1, 2]
    r20, r21, r22 = rmat[2, 0], rmat[2, 1], rmat[2, 2]

    rho = jnp.array([r21 - r12, r02 - r20, r10 - r01])
    sin_theta = jnp.linalg.norm(rho) / 2
    cos_theta = jnp.clip((r00 + r11 + r22 - 1) / 2, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)
    near_pi = sin_theta < _NEAR_PI_SIN_THRESH

    rvec = rho * (theta / jnp.where(near_pi, 1.0, 2.0 * sin_theta))

    ax = jnp.sqrt(jnp.maximum((r00 + 1.0) / 2, 0.0))
    ay = jnp.sqrt(jnp.maximum((r11 + 1.0) / 2, 0.0)) * jnp.where(r01 < 0, -1.0, 1.0)
    az = jnp.sqrt(jnp.maximum((r22 + 1.0) / 2, 0.0)) * jnp.where(r02 < 0, -1.0, 1.0)
    flip = (
        (jnp.abs(ax) < jnp.abs(ay))
        & (jnp.abs(ax) < jnp.abs(az))
        & ((r12 > 0) != (ay * az > 0))
    )
    az = jnp.where(flip, -az, az)
    axis = jnp.array([ax, ay, az])
    axis_norm = jnp.linalg.norm(axis)
    rvec_pi = axis * (theta / jnp.where(axis_norm == 0, 1.0, axis_norm))
    rvec_degenerate = jnp.where(cos_theta > 0, 0.0, rvec_pi)

    return jnp.where(near_pi, rvec_degenerate, rvec)


def distort_one(
    xy: Float[Array, "2"],
    dist: Float[Array, "K"],
) -> Float[Array, "2"]:
    """Distortion model applied to a single 2D point.

    Single-instance variant of :func:`distort`.
    """
    n = dist.shape[-1]
    if n == 0:
        return xy
    x, y = xy[0], xy[1]
    x2, y2 = x * x, y * y
    r2 = x2 + y2

    num = 1.0 + dist[0] * r2
    r4 = r6 = None
    if n >= 2:
        r4 = r2 * r2
        num = num + dist[1] * r4
    if n >= 5:
        r6 = r4 * r2
        num = num + dist[4] * r6
    den = 1.0
    if n >= 6:
        den = den + dist[5] * r2
    if n >= 7:
        den = den + dist[6] * r4
    if n >= 8:
        den = den + dist[7] * r6
    mult = num / den

    add_x = jnp.zeros(())
    add_y = jnp.zeros(())
    if n >= 3:
        xy_prod = x * y
        add_x = 2 * dist[2] * xy_prod
        add_y = dist[2] * (r2 + 2 * y2)
    if n >= 4:
        add_x = add_x + dist[3] * (r2 + 2 * x2)
        add_y = add_y + 2 * dist[3] * xy_prod
    if n >= 9:
        add_x = add_x + dist[8] * r2
    if n >= 10:
        add_x = add_x + dist[9] * r4
    if n >= 11:
        add_y = add_y + dist[10] * r2
    if n >= 12:
        add_y = add_y + dist[11] * r4

    return jnp.stack([x * mult + add_x, y * mult + add_y])


def project_full_one(
    pt3d: Float[Array, "3"],
    rvec: Float[Array, "3"],
    tvec: Float[Array, "3"],
    intr: Float[Array, "P"],
    dist: Float[Array, "K"],
) -> Float[Array, "2"]:
    """Project a single 3D point through a single camera.

    Single-instance variant of :func:`project_full` designed to be composed
    with :func:`jax.vmap` over observations and :func:`jax.jacfwd` over the
    camera parameters and the 3D point.

    Argument order matches :func:`project_full`: operand (``pt3d``) first,
    then camera parameters in the canonical order
    ``rvec, tvec, intr, dist``.
    """
    rmat = rvec_to_rmat_one(rvec)
    p_cam = rmat @ pt3d + tvec
    xy = p_cam[:2] / p_cam[2]
    xy = distort_one(xy, dist)
    fx, fy = intr[0], intr[-3]
    cx, cy = intr[-2], intr[-1]
    return jnp.stack([fx * xy[0] + cx, fy * xy[1] + cy])


# -- batched / composed functions -------------------------------------------


@cpu_jit
def intr_to_kmat(
    intr: Float[Array, "*batch P"],
) -> Float[Array, "*batch 3 3"]:
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
    kmat = jnp.zeros((*intr.shape[:-1], 3, 3))
    kmat = kmat.at[..., 0, 0].set(intr[..., 0])
    kmat = kmat.at[..., 1, 1].set(intr[..., -3])
    kmat = kmat.at[..., :2, 2].set(intr[..., -2:])
    kmat = kmat.at[..., 2, 2].set(1.0)
    return kmat


@cpu_jit
def rvec_to_rmat(
    rvec: Float[Array, "*batch 3"],
) -> Float[Array, "*batch 3 3"]:
    """Convert axis-angle rotation vectors to rotation matrices (Rodrigues).

    Implements ``R = I + a * W + b * W^2`` with ``W = skew(rvec)``,
    ``a = sin(theta) / theta`` and ``b = (1 - cos(theta)) / theta^2``. Working
    on the unnormalized axis avoids ``0/0`` at ``theta = 0``; evaluating ``a``
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
    flat = rvec.reshape(-1, 3)
    out = jax.vmap(rvec_to_rmat_one)(flat)
    return out.reshape(*rvec.shape[:-1], 3, 3)


@cpu_jit
def rmat_to_rvec(
    rmat: Float[Array, "*batch 3 3"],
) -> Float[Array, "*batch 3"]:
    """Convert rotation matrices to axis-angle rotation vectors.

    Vectorized port of OpenCV's ``Rodrigues`` (matrix -> vector). The axis is
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
    flat = rmat.reshape(-1, 3, 3)
    out = jax.vmap(rmat_to_rvec_one)(flat)
    return out.reshape(*rmat.shape[:-2], 3)


@cpu_jit
def project_pmat(
    pts3d: Float[Array, "*pts 3"],
    pmats: Float[Array, "*cams 3 4"],
) -> Float[Array, "*cams *pts 2"]:
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
    pmats_flat = pmats.reshape(-1, 3, 4)
    pts_flat = pts3d.reshape(-1, 3).T
    pts2dh = pmats_flat[:, :, :3] @ pts_flat + pmats_flat[:, :, 3:]
    pts2d = (pts2dh[:, :2] / pts2dh[:, 2:]).transpose(0, 2, 1)
    return pts2d.reshape(output_shape)


@cpu_jit
def triangulate_dlt(
    pts2d: Float[Array, "V *pts 2"],
    pmats: Float[Array, "V 3 4"],
) -> Float[Array, "*pts 3"]:
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
    a = jnp.einsum("v...i,vj->...vij", pts2d, pmats[:, 2]) - pmats[:, :2]
    valid = jnp.moveaxis(jnp.isfinite(pts2d).all(axis=-1), 0, -1)
    a = jnp.where(valid[..., None, None], a, 0.0)
    a = a.reshape((*a.shape[:-3], -1, 4))
    ata = jnp.einsum("...ij,...ik->...jk", a, a)
    pts3dh = jnp.linalg.eigh(ata)[1][..., :, 0]
    pts3d = pts3dh[..., :3] / pts3dh[..., 3:]
    return jnp.where((valid.sum(axis=-1) < 2)[..., None], jnp.nan, pts3d)


@cpu_jit
def distort(
    pts2d: Float[Array, "V *pts 2"],
    dists: Float[Array, "V K"],
) -> Float[Array, "V *pts 2"]:
    """Apply OpenCV-style radial + tangential + thin-prism distortion.

    For normalized image coordinates ``(x, y)`` with ``r^2 = x^2 + y^2``, the
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
        Normalized 2D coordinates of shape ``(V, *pts, 2)``.
    dists
        Distortion coefficients of shape ``(V, K)`` with ``K`` in
        ``{0, 1, ..., 12}``.

    Returns
    -------
    Distorted 2D coordinates of shape ``(V, *pts, 2)``.
    """
    if dists.shape[-1] == 0:
        return pts2d
    v = pts2d.shape[0]
    flat = pts2d.reshape(v, -1, 2)
    out = jax.vmap(jax.vmap(distort_one, in_axes=(0, None)), in_axes=(0, 0))(
        flat, dists
    )
    return out.reshape(pts2d.shape)


@cpu_jit
def project_full(
    pts3d: Float[Array, "*pts 3"],
    rvecs: Float[Array, "V 3"],
    tvecs: Float[Array, "V 3"],
    intrs: Float[Array, "V P"] | Float[Array, "P"],
    dists: Float[Array, "V K"] | Float[Array, "K"],
) -> Float[Array, "V *pts 2"]:
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
    v = rvecs.shape[0]
    pts_shape = pts3d.shape[:-1]
    pts_flat = pts3d.reshape(-1, 3)
    if intrs.ndim == 1:
        intrs = jnp.broadcast_to(intrs, (v, intrs.shape[0]))
    if dists.ndim == 1:
        dists = jnp.broadcast_to(dists, (v, dists.shape[0]))
    project_v = jax.vmap(
        jax.vmap(project_full_one, in_axes=(0, None, None, None, None)),
        in_axes=(None, 0, 0, 0, 0),
    )
    out = project_v(pts_flat, rvecs, tvecs, intrs, dists)
    return out.reshape(v, *pts_shape, 2)
