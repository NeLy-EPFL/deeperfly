"""Packed parameter state for bundle adjustment (backend-agnostic).

The state vector concatenates ``[rvecs, tvecs, intrs, dists, pts3d]`` flat.
Index arrays (``rvecs_idx``, etc.) recover the original shapes; a boolean
``fixed`` mask marks parameters that are held constant during optimisation.
``intrs`` and ``dists`` are broadcast across views by default (shared camera
model), but the indexing scheme is general enough to allow per-view freedom by
flipping entries in ``fixed``.

The same packed state is consumed by both the NumPy/SciPy solver in
:mod:`deeperfly.np.bundle_adjustment` and the JAX solvers in
:mod:`deeperfly.jax.bundle_adjustment`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from jaxtyping import Bool, Float, Int

from .np.geometry import intr_to_kmat, rvec_to_rmat, triangulate_dlt


class BAState(NamedTuple):
    """Packed bundle-adjustment state ready for ``bundle_adjust``.

    ``values`` is the flat parameter vector; ``fixed`` is a boolean mask of
    parameters to hold constant. The ``*_idx`` arrays index into ``values`` to
    recover the original ``rvecs`` / ``tvecs`` / ``intrs`` / ``dists`` /
    ``pts3d`` arrays. ``pts2d`` is the observation tensor.
    """

    values: Float[np.ndarray, "n_params"]
    fixed: Bool[np.ndarray, "n_params"]
    rvecs_idx: Int[np.ndarray, "V 3"]
    tvecs_idx: Int[np.ndarray, "V 3"]
    intrs_idx: Int[np.ndarray, "V P"]
    dists_idx: Int[np.ndarray, "V K"]
    pts3d_idx: Int[np.ndarray, "N 3"]
    pts2d: Float[np.ndarray, "V N 2"]


def initialize_pts3d(
    pts2d: Float[np.ndarray, "V *pts 2"],
    rvecs: Float[np.ndarray, "V 3"],
    tvecs: Float[np.ndarray, "V 3"],
    intrs: Float[np.ndarray, "V P"] | Float[np.ndarray, "P"],
) -> Float[np.ndarray, "*pts 3"]:
    """Triangulate initial 3D points from 2D observations and camera poses."""
    kmat = intr_to_kmat(intrs)
    rtmat = np.concatenate((rvec_to_rmat(rvecs), tvecs[..., None]), axis=-1)
    return triangulate_dlt(pts2d, kmat @ rtmat)


def prep_args(
    pts2d: Float[np.ndarray, "V N 2"],
    rvecs: Float[np.ndarray, "V 3"],
    tvecs: Float[np.ndarray, "V 3"],
    intrs: Float[np.ndarray, "V P"] | Float[np.ndarray, "P"],
    dists: Float[np.ndarray, "V K"] | Float[np.ndarray, "K"],
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
        Initial intrinsics / distortion, shared across views.

    Returns
    -------
    A :class:`BAState` ready to splat into ``bundle_adjust``.
    """
    pts3d = initialize_pts3d(pts2d, rvecs, tvecs, intrs)
    arrs = [rvecs, tvecs, intrs, dists, pts3d]
    values = np.concatenate([a.ravel() for a in arrs])
    size_cumsum = [0, *np.cumsum([a.size for a in arrs])]
    fixed = np.zeros_like(values, dtype=bool)
    # Fix intrinsics and distortions by default.
    fixed[size_cumsum[2] : size_cumsum[4]] = True
    rvecs_idx, tvecs_idx, intrs_idx, dists_idx, pts3d_idx = (
        np.arange(a.size).reshape(a.shape) + size_cumsum[i] for i, a in enumerate(arrs)
    )
    # Broadcast shared intrinsics / distortions to all V cameras.
    n_views = len(rvecs)
    intrs_idx = np.stack((intrs_idx,) * n_views, axis=0)
    dists_idx = np.stack((dists_idx,) * n_views, axis=0)
    return BAState(
        values, fixed, rvecs_idx, tvecs_idx, intrs_idx, dists_idx, pts3d_idx, pts2d
    )
