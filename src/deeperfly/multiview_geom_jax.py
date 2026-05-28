"""Multi-view geometry primitives (JAX).

JAX equivalents of :mod:`deeperfly.multiview_geom`. See that module for
conventions and the derivations of the rotation-conversion numerics. All
functions here are JIT- and grad-friendly.

The ``*_one`` variants operate on a single observation (no leading batch axes)
and are designed to be composed with :func:`jax.vmap` and :func:`jax.jacfwd`
for bundle adjustment.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

jax.config.update("jax_enable_x64", True)

# See multiview_geom.py for the meaning of these thresholds.
_SMALL_THETA_SQ = 1e-8
_NEAR_PI_SIN_THRESH = 1e-5


def intr_to_kmat(
    intr: Float[Array, "*batch P"],
) -> Float[Array, "*batch 3 3"]:
    """Build 3x3 camera intrinsic matrices from packed intrinsics.

    See :func:`deeperfly.multiview_geom.intr_to_kmat`.
    """
    kmat = jnp.zeros((*intr.shape[:-1], 3, 3))
    kmat = kmat.at[..., 0, 0].set(intr[..., 0])
    kmat = kmat.at[..., 1, 1].set(intr[..., -3])
    kmat = kmat.at[..., :2, 2].set(intr[..., -2:])
    kmat = kmat.at[..., 2, 2].set(1.0)
    return kmat


def rvec_to_rmat(
    rvec: Float[Array, "*batch 3"],
) -> Float[Array, "*batch 3 3"]:
    """Convert axis-angle rotation vectors to rotation matrices (Rodrigues).

    See :func:`deeperfly.multiview_geom.rvec_to_rmat`.
    """
    theta_sq = jnp.einsum("...i,...i->...", rvec, rvec)
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
    zero = jnp.zeros_like(rvec[..., 0])
    skew = jnp.stack(
        [
            jnp.stack([zero, -rvec[..., 2], rvec[..., 1]], axis=-1),
            jnp.stack([rvec[..., 2], zero, -rvec[..., 0]], axis=-1),
            jnp.stack([-rvec[..., 1], rvec[..., 0], zero], axis=-1),
        ],
        axis=-2,
    )
    outer = rvec[..., :, None] * rvec[..., None, :]
    diag = (1 - cosc_theta * theta_sq)[..., None, None] * jnp.eye(3)
    return (
        diag + sinc_theta[..., None, None] * skew + cosc_theta[..., None, None] * outer
    )


def rmat_to_rvec(
    rmat: Float[Array, "*batch 3 3"],
) -> Float[Array, "*batch 3"]:
    """Convert rotation matrices to axis-angle rotation vectors.

    See :func:`deeperfly.multiview_geom.rmat_to_rvec`.
    """
    r00, r01, r02 = rmat[..., 0, 0], rmat[..., 0, 1], rmat[..., 0, 2]
    r10, r11, r12 = rmat[..., 1, 0], rmat[..., 1, 1], rmat[..., 1, 2]
    r20, r21, r22 = rmat[..., 2, 0], rmat[..., 2, 1], rmat[..., 2, 2]

    rho = jnp.stack((r21 - r12, r02 - r20, r10 - r01), axis=-1)
    sin_theta = jnp.linalg.norm(rho, axis=-1) / 2
    cos_theta = jnp.clip((r00 + r11 + r22 - 1) / 2, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)
    near_pi = sin_theta < _NEAR_PI_SIN_THRESH

    rvec = rho * (theta / jnp.where(near_pi, 1.0, 2.0 * sin_theta))[..., None]

    ax = jnp.sqrt(jnp.maximum((r00 + 1.0) / 2, 0.0))
    ay = jnp.sqrt(jnp.maximum((r11 + 1.0) / 2, 0.0)) * jnp.where(r01 < 0, -1.0, 1.0)
    az = jnp.sqrt(jnp.maximum((r22 + 1.0) / 2, 0.0)) * jnp.where(r02 < 0, -1.0, 1.0)
    flip = (
        (jnp.abs(ax) < jnp.abs(ay))
        & (jnp.abs(ax) < jnp.abs(az))
        & ((r12 > 0) != (ay * az > 0))
    )
    az = jnp.where(flip, -az, az)
    axis = jnp.stack((ax, ay, az), axis=-1)
    axis_norm = jnp.linalg.norm(axis, axis=-1)
    rvec_pi = axis * (theta / jnp.where(axis_norm == 0, 1.0, axis_norm))[..., None]
    rvec_degenerate = jnp.where((cos_theta > 0)[..., None], 0.0, rvec_pi)

    return jnp.where(near_pi[..., None], rvec_degenerate, rvec)


def project(
    pts3d: Float[Array, "*pts 3"],
    pmats: Float[Array, "*cams 3 4"],
) -> Float[Array, "*cams *pts 2"]:
    """Project 3D world points to 2D image points using 3x4 projection matrices.

    See :func:`deeperfly.multiview_geom.project`.
    """
    output_shape = (*pmats.shape[:-2], *pts3d.shape[:-1], 2)
    pmats_flat = pmats.reshape(-1, 3, 4)
    pts_flat = pts3d.reshape(-1, 3).T
    pts2dh = pmats_flat[:, :, :3] @ pts_flat + pmats_flat[:, :, 3:]
    pts2d = (pts2dh[:, :2] / pts2dh[:, 2:]).transpose(0, 2, 1)
    return pts2d.reshape(output_shape)


def triangulate_dlt(
    pts2d: Float[Array, "V *pts 2"],
    pmats: Float[Array, "V 3 4"],
) -> Float[Array, "*pts 3"]:
    """Triangulate 3D points by direct linear transformation (DLT).

    See :func:`deeperfly.multiview_geom.triangulate_dlt`.
    """
    a = jnp.einsum("v...i,vj->...vij", pts2d, pmats[:, 2]) - pmats[:, :2]
    valid = jnp.moveaxis(jnp.isfinite(pts2d).all(axis=-1), 0, -1)
    a = jnp.where(valid[..., None, None], a, 0.0)
    a = a.reshape((*a.shape[:-3], -1, 4))
    ata = jnp.einsum("...ij,...ik->...jk", a, a)
    pts3dh = jnp.linalg.eigh(ata)[1][..., :, 0]
    pts3d = pts3dh[..., :3] / pts3dh[..., 3:]
    return jnp.where((valid.sum(axis=-1) < 2)[..., None], jnp.nan, pts3d)


def distort(
    pts2d: Float[Array, "V *pts 2"],
    dists: Float[Array, "V K"],
) -> Float[Array, "V *pts 2"]:
    """Apply OpenCV-style radial + tangential + thin-prism distortion.

    See :func:`deeperfly.multiview_geom.distort`.
    """
    n = dists.shape[-1]
    if n == 0:
        return pts2d
    coef_shape = (dists.shape[0],) + (1,) * (pts2d.ndim - 2)
    k = [dists[:, i].reshape(coef_shape) for i in range(n)]
    x, y = pts2d[..., 0], pts2d[..., 1]
    x2, y2 = x * x, y * y
    r2 = x2 + y2

    num = 1.0 + k[0] * r2
    r4 = r6 = None
    if n >= 2:
        r4 = r2 * r2
        num = num + k[1] * r4
    if n >= 5:
        r6 = r4 * r2
        num = num + k[4] * r6
    den = 1.0
    if n >= 6:
        den = den + k[5] * r2
    if n >= 7:
        den = den + k[6] * r4
    if n >= 8:
        den = den + k[7] * r6
    mult = num / den

    add_x = jnp.zeros_like(x)
    add_y = jnp.zeros_like(y)
    if n >= 3:
        xy = x * y
        add_x = 2 * k[2] * xy
        add_y = k[2] * (r2 + 2 * y2)
    if n >= 4:
        add_x = add_x + k[3] * (r2 + 2 * x2)
        add_y = add_y + 2 * k[3] * xy
    if n >= 9:
        add_x = add_x + k[8] * r2
    if n >= 10:
        add_x = add_x + k[9] * r4
    if n >= 11:
        add_y = add_y + k[10] * r2
    if n >= 12:
        add_y = add_y + k[11] * r4

    return jnp.stack([x * mult + add_x, y * mult + add_y], axis=-1)


def project_full(
    pts3d: Float[Array, "*pts 3"],
    rvecs: Float[Array, "V 3"],
    tvecs: Float[Array, "V 3"],
    intrs: Float[Array, "V P"] | Float[Array, "P"],
    dists: Float[Array, "V K"] | Float[Array, "K"],
) -> Float[Array, "V *pts 2"]:
    """Project 3D world points through full camera models.

    See :func:`deeperfly.multiview_geom.project_full`.
    """
    pts_dims = pts3d.shape[:-1]
    rmats = rvec_to_rmat(rvecs)
    xyz = ((rmats @ pts3d.T) + tvecs[..., None]).transpose(0, 2, 1)
    xy = xyz[:, :, :2] / xyz[:, :, 2:3]
    xy = distort(xy, dists)
    intr_shape = (*intrs.shape[:-1], *(1,) * len(pts_dims), 2)
    return xy * intrs[..., jnp.array([0, -3])].reshape(intr_shape) + intrs[
        ..., -2:
    ].reshape(intr_shape)


# -- per-observation primitives (for vmap / jacfwd in bundle adjustment) -----


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


def project_one(
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
