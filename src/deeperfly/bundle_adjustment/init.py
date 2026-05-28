"""Packed parameter state for bundle adjustment (backend-agnostic).

The state vector concatenates ``[rvecs, tvecs, intrs, dists, pts3d]`` flat.
Index arrays (``rvecs_idx``, etc.) recover the original shapes; a boolean
``fixed`` mask marks parameters that are held constant during optimisation.
``intrs`` and ``dists`` are broadcast across views by default (shared camera
model), but the indexing scheme is general enough to allow per-view freedom by
flipping entries in ``fixed``.

The same packed state is consumed by the JAX solvers in
:mod:`deeperfly.jax.bundle_adjustment`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from ..geometry import intr_to_kmat, rvec_to_rmat, triangulate_dlt


class BAState(NamedTuple):
    """Packed bundle-adjustment state ready for ``bundle_adjust``.

    ``values`` is the flat parameter vector; ``fixed`` is a boolean mask of
    parameters to hold constant. The ``*_idx`` arrays index into ``values`` to
    recover the original ``rvecs`` / ``tvecs`` / ``intrs`` / ``dists`` /
    ``pts3d`` arrays. ``pts2d`` is the observation tensor.
    """

    values: Float[Array, "n_params"]
    fixed: Bool[Array, "n_params"]
    rvecs_idx: Int[Array, "V 3"]
    tvecs_idx: Int[Array, "V 3"]
    intrs_idx: Int[Array, "V P"]
    dists_idx: Int[Array, "V K"]
    pts3d_idx: Int[Array, "N 3"]
    pts2d: Float[Array, "V N 2"]


def initialize_pts3d(
    pts2d: Float[Array, "V *pts 2"],
    rvecs: Float[Array, "V 3"],
    tvecs: Float[Array, "V 3"],
    intrs: Float[Array, "V P"] | Float[Array, "P"],
) -> Float[Array, "*pts 3"]:
    """Triangulate initial 3D points from 2D observations and camera poses."""
    kmat = intr_to_kmat(intrs)
    rtmat = jnp.concatenate(
        (jnp.asarray(rvec_to_rmat(rvecs)), tvecs[..., None]), axis=-1
    )
    return triangulate_dlt(pts2d, kmat @ rtmat)


def prep_args(
    pts2d: Float[Array, "cams pts 2"],
    rvecs: Float[Array, "cams 3"],
    tvecs: Float[Array, "cams 3"],
    intrs: Float[Array, "cams P"] | Float[Array, "P"],
    dists: Float[Array, "cams K"] | Float[Array, "K"] | None = None,
    pts3d: Float[Array, "N 3"] | None = None,
    fix_rvecs: Bool[Array, "cams"] | Bool[Array, "cams 3"] | int | bool = False,
    fix_tvecs: Bool[Array, "cams"] | Bool[Array, "cams 3"] | int | bool = False,
    fix_intrs: Bool[Array, "cams"] | Bool[Array, "cams P"] | int | bool = True,
    fix_dists: Bool[Array, "cams"] | Bool[Array, "cams K"] | int | bool = True,
) -> BAState:
    """Build a :class:`BAState` from observations and initial camera parameters.

    The 3D points are initialised by triangulating ``pts2d`` from the given
    cameras. Intrinsics and distortion are broadcast to every view and then
    marked fixed in the returned ``fixed`` mask, so the optimiser solves only
    for extrinsics and 3D points by default. Flip the corresponding entries of
    ``fixed`` to free them.

    Parameters
    ----------
    pts2d
        Observed 2D points of shape ``(V, N, 2)`` with NaNs for missing.
    rvecs, tvecs
        Initial per-view extrinsics of shape ``(V, 3)``.
    intrs, dists
        Initial intrinsics / distortion of shape ``(V, P)`` and ``(V, K)`` respectively,
        or ``(P,)`` and ``(K,)`` respectively if shared across views.
    fix_rvecs, fix_tvecs, fix_intrs, fix_dists
        Which parameters to hold constant. Accepts:

        - ``bool``: fix all (or none) of this parameter group.
        - ``int``: fix the parameters of camera at that index (e.g.
          ``fix_rvecs=0`` and ``fix_tvecs=0`` anchors camera 0).
        - ``Bool[Array, "cams"]``: fix per-camera (broadcast across the
          parameter dim).
        - ``Bool[Array, "cams P"]`` / ``Bool[Array, "P"]``: a full mask matching
          the underlying parameter array.

    Returns
    -------
    A :class:`BAState` ready to splat into ``bundle_adjust``.
    """
    if pts3d is None:
        pts3d = initialize_pts3d(pts2d, rvecs, tvecs, intrs)
    if dists is None:
        dists = jnp.zeros((0,))

    n_views = len(rvecs)
    rvecs = np.asarray(rvecs)
    tvecs = np.asarray(tvecs)
    intrs = np.asarray(intrs)
    dists = np.asarray(dists)
    pts3d = np.asarray(pts3d)
    pts2d = np.asarray(pts2d)

    arrs = [rvecs, tvecs, intrs, dists, pts3d]
    values = np.concatenate([a.ravel() for a in arrs])
    offsets = np.cumsum([0, *(a.size for a in arrs)])

    rvecs_idx = np.arange(rvecs.size).reshape(rvecs.shape) + offsets[0]
    tvecs_idx = np.arange(tvecs.size).reshape(tvecs.shape) + offsets[1]
    intrs_idx = np.arange(intrs.size).reshape(intrs.shape) + offsets[2]
    dists_idx = np.arange(dists.size).reshape(dists.shape) + offsets[3]
    pts3d_idx = np.arange(pts3d.size).reshape(pts3d.shape) + offsets[4]

    if intrs_idx.ndim == 1:
        intrs_idx = np.broadcast_to(intrs_idx, (n_views, *intrs_idx.shape))
    if dists_idx.ndim == 1:
        dists_idx = np.broadcast_to(dists_idx, (n_views, *dists_idx.shape))

    fixed = np.zeros(values.size, dtype=bool)
    fixed[offsets[0] : offsets[1]] = _expand_fix(fix_rvecs, rvecs).ravel()
    fixed[offsets[1] : offsets[2]] = _expand_fix(fix_tvecs, tvecs).ravel()
    fixed[offsets[2] : offsets[3]] = _expand_fix(fix_intrs, intrs).ravel()
    fixed[offsets[3] : offsets[4]] = _expand_fix(fix_dists, dists).ravel()

    return BAState(
        values, fixed, rvecs_idx, tvecs_idx, intrs_idx, dists_idx, pts3d_idx, pts2d
    )


def _expand_fix(fix, arr: np.ndarray) -> np.ndarray:
    """Expand a ``fix_*`` spec to a bool mask matching ``arr.shape``."""
    # isinstance(True, int) is True, so check bool first.
    if isinstance(fix, bool):
        return np.full(arr.shape, fix, dtype=bool)
    if isinstance(fix, int):
        mask = np.zeros(arr.shape, dtype=bool)
        mask[fix] = True
        return mask
    fix = np.asarray(fix, dtype=bool)
    if fix.shape == arr.shape:
        return fix
    if fix.ndim == 1 and arr.ndim > 1 and fix.shape[0] == arr.shape[0]:
        return np.broadcast_to(fix[:, None], arr.shape)
    raise ValueError(
        f"fix mask shape {fix.shape} incompatible with parameter shape {arr.shape}"
    )
