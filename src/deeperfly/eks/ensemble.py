"""Ensembling and the cross-view variance inflation that guards the smoother.

Two of the three things the "ensemble Kalman smoother" name promises live here.
The first is the ensemble itself: several independently-trained detectors give
each observation a *center* and a *spread*, and the spread is the observation
noise the filter needs -- uncertainty measured rather than assumed. The second is
the variance inflation Lightning Pose 3D adds: before smoothing, test each view's
prediction against what the *other* views say, and where it disagrees beyond
chance, inflate its variance so the filter down-weights it instead of following
it.

The inflation is the part that works at ensemble size one. It needs two views, not
two models -- so on a calibrated rig with a single detector it is fully active,
and it is what stops the smoother from tracking a blown detection into the wrong
place.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Bool, Float

from ..rig.cameras import CameraGroup
from .core import project_all

__all__ = ["ensemble_statistics", "stack_views", "unstack_views", "inflate_variances"]

#: Variance assigned to a cell whose ensemble statistics come out undefined. Large
#: enough that the filter ignores the observation; finite so nothing turns to NaN.
_UNDEFINED_VAR = 1e3

#: Floor on a mean confidence before it is inverted into a variance.
_MIN_CONF = 1e-5

#: Ridge added to the cross-view information matrix, relative to its own trace.
_RIDGE = 1e-8

#: Guard against dividing by a zero variance in the whitening below.
_EPS = 1e-6


def ensemble_statistics(
    pts2d: Float[np.ndarray, "M V T P 2"],
    conf: Float[np.ndarray, "M V T P"] | None = None,
    *,
    avg_mode: str = "median",
    var_mode: str = "confidence_weighted_var",
) -> tuple[Float[np.ndarray, "V T P 2"], Float[np.ndarray, "V T P 2"]]:
    """Collapse an ensemble of detectors into a center and a per-axis variance.

    Parameters
    ----------
    pts2d
        Per-model 2D detections, ``(M, V, T, P, 2)``. NaN marks an unobserved
        ``(view, point)`` cell.
    conf
        Per-model confidences ``(M, V, T, P)``, or ``None`` to treat every
        observation as equally confident.
    avg_mode
        ``"median"`` (the default, and the reference implementation's) or
        ``"mean"``. The median is what makes the *center* robust to one member
        blowing up; the variance still notices that it did.
    var_mode
        ``"var"`` for the plain across-model variance, or
        ``"confidence_weighted_var"`` (default) to divide it by the mean
        confidence, so a cell every model is unsure about is treated as noisier
        than its agreement alone suggests.

    Returns
    -------
    center : Float[np.ndarray, "V T P 2"]
        The ensemble consensus 2D.
    var : Float[np.ndarray, "V T P 2"]
        Its per-axis variance in px^2.

    Notes
    -----
    With a single member (``M == 1``) the spread is identically zero, so the
    variance falls back to ``1 / confidence`` -- the same fallback the reference
    implementation makes. That is a *prior*, not a calibrated uncertainty: it
    ranks observations sensibly but its absolute scale is arbitrary, which is
    only harmless because the fitted smoothing parameter rescales the process
    noise against it. Running a real ensemble is what turns this term into a
    measurement.
    """
    pts2d = np.asarray(pts2d, dtype=float)
    if pts2d.ndim != 5:
        raise ValueError(f"pts2d must be (M, V, T, P, 2); got {pts2d.shape}")
    if avg_mode not in ("median", "mean"):
        raise ValueError(f"avg_mode must be 'median' or 'mean'; got {avg_mode!r}")
    if var_mode not in ("var", "confidence_weighted_var"):
        raise ValueError(
            f"var_mode must be 'var' or 'confidence_weighted_var'; got {var_mode!r}"
        )
    n_models = pts2d.shape[0]

    with warnings.catch_warnings():  # all-NaN cells are expected (unobserved)
        warnings.simplefilter("ignore", RuntimeWarning)
        center = (np.nanmedian if avg_mode == "median" else np.nanmean)(pts2d, axis=0)
        if conf is None:
            mean_conf = np.ones(pts2d.shape[1:4], dtype=float)
        else:
            mean_conf = np.asarray(conf, dtype=float).mean(axis=0)
        if n_models == 1:
            single = 1.0 / np.maximum(mean_conf, _MIN_CONF)
            var = np.repeat(single[..., None], 2, axis=-1)
        else:
            var = np.nanvar(pts2d, axis=0)
            if var_mode == "confidence_weighted_var":
                var = var / np.maximum(mean_conf, _MIN_CONF)[..., None]

    var = np.where(np.isfinite(var), var, _UNDEFINED_VAR)
    return center, var


def stack_views(arr: np.ndarray) -> np.ndarray:
    """``(V, T, P, D) -> (T, P, V * D)``, view-major -- the smoother's layout."""
    v, t, p, d = arr.shape
    return np.transpose(arr, (1, 2, 0, 3)).reshape(t, p, v * d)


def unstack_views(arr: np.ndarray, n_views: int) -> np.ndarray:
    """Inverse of :func:`stack_views`: ``(T, P, V * D) -> (V, T, P, D)``."""
    t, p, vd = arr.shape
    return np.transpose(arr.reshape(t, p, n_views, vd // n_views), (2, 0, 1, 3))


def _cell_inflation(
    x: Float[jnp.ndarray, "2V"],
    var: Float[jnp.ndarray, "2V"],
    mask: Bool[jnp.ndarray, "2V"],
    pt3d: Float[jnp.ndarray, "3"],
    testable: Bool[jnp.ndarray, ""],
    cams: tuple,
    threshold: float,
    factor: float,
    n_rounds: int,
) -> Float[jnp.ndarray, "2V"]:
    """Inflate one cell's per-view variances until no view disagrees beyond threshold.

    Implements the paper's test ``d = (x - x_hat)^T Q^{-1} (x - x_hat)`` with
    ``Q = D + W B W^T``, where ``x_hat`` is what the *other* views predict for this
    one under an uninformative prior on the latent. The reference implementation
    learns ``W`` with factor analysis because it must also serve an uncalibrated,
    PCA-based path; here the rig is known, so ``W`` is the exact projection
    Jacobian at the current 3D estimate and ``B = (W^T D^{-1} W)^{-1}`` is the
    corresponding posterior covariance. Same formula, no fitted approximation --
    and no observations spent estimating one.
    """

    def h(z):
        return project_all(z, *cams)

    y_hat = h(pt3d)
    jac = jax.jacfwd(h)(pt3d) * mask[:, None]  # (2V, 3) -- W
    delta_x = jnp.where(mask, x - y_hat, 0.0)
    view_ok = jnp.all(mask.reshape(-1, 2), axis=-1)  # (V,)
    n_views = jnp.sum(view_ok)

    def body(_, v):
        prec = jnp.where(mask, 1.0 / (v + _EPS), 0.0)  # diag(D^-1)
        info = jac.T @ (jac * prec[:, None])  # W^T D^-1 W
        ridge = _RIDGE * jnp.maximum(jnp.trace(info) / 3.0, 1e-12)
        post = jnp.linalg.inv(info + ridge * jnp.eye(3))  # B
        delta_z = post @ (jac.T @ (prec * delta_x))
        resid = jnp.where(mask, delta_x - jac @ delta_z, 0.0).reshape(-1, 2)  # (V, 2)

        jac_v = jac.reshape(-1, 2, 3)
        # Q_v = diag(v_view) + W_view B W_view^T
        pred_cov = jnp.einsum("vij,jk,vlk->vil", jac_v, post, jac_v)
        pred_cov = pred_cov + jnp.eye(2) * v.reshape(-1, 2)[:, None, :]
        whitened = jnp.linalg.solve(pred_cov, resid[..., None])[..., 0]
        maha = jnp.sum(resid * whitened, axis=-1)  # (V,)

        exceeds = view_ok & testable & (maha > threshold)
        # With exactly two views the residual cannot say which of them is wrong,
        # so inflate both rather than guess (the reference does the same).
        exceeds = jnp.where(n_views == 2, view_ok & jnp.any(exceeds), exceeds)
        return v * jnp.repeat(jnp.where(exceeds, factor, 1.0), 2)

    return lax.fori_loop(0, n_rounds, body, var)


def inflate_variances(
    center: Float[np.ndarray, "V T P 2"],
    var: Float[np.ndarray, "V T P 2"],
    mask: Bool[np.ndarray, "V T P"],
    pts3d: Float[np.ndarray, "T P 3"],
    cameras: CameraGroup,
    *,
    threshold: float = 5.0,
    factor: float = 10.0,
    n_rounds: int = 8,
) -> tuple[Float[np.ndarray, "V T P 2"], int]:
    """Inflate the variance of every view that disagrees with its cross-view consensus.

    Parameters
    ----------
    center, var
        The ensemble consensus 2D and its variance (see
        :func:`ensemble_statistics`).
    mask
        ``(V, T, P)``, ``True`` where the cell is observed.
    pts3d
        The 3D estimate the projection is linearized about, ``(T, P, 3)``. Cells
        that are not finite are skipped rather than tested.
    cameras
        The rig.
    threshold
        Mahalanobis distance above which a view is called inconsistent. The
        paper's example value is 5.
    factor
        Multiplier applied to an offending view's variance each round. The paper
        describes doubling; the reference implementation ships 10, which is what
        the ``eks multicam`` CLI actually runs, so 10 is the default here.
    n_rounds
        How many times to re-test and re-inflate. Each round compounds, so the
        default admits up to a 1e8 inflation.

    Returns
    -------
    var : Float[np.ndarray, "V T P 2"]
        The inflated variances.
    n_inflated : int
        How many ``(view, frame, point)`` observations were down-weighted -- a
        useful sanity number: a run that inflates almost nothing is telling you
        the views already agree.
    """
    n_views, n_frames, n_points = mask.shape
    finite3d = np.isfinite(np.asarray(pts3d, dtype=float)).all(axis=-1)  # (T, P)
    # A non-finite linearization point cannot be tested; substitute a harmless
    # placeholder so no NaN enters the jitted kernel, and gate it off instead.
    with warnings.catch_warnings():  # a never-triangulated point is all-NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        fallback = np.nanmedian(np.asarray(pts3d, dtype=float), axis=0)  # (P, 3)
    fallback = np.where(np.isfinite(fallback), fallback, 0.0)
    safe3d = np.where(finite3d[..., None], pts3d, fallback[None])

    x = stack_views(center)  # (T, P, 2V)
    v = stack_views(var)
    m = np.repeat(np.transpose(mask, (1, 2, 0)), 2, axis=-1)  # (T, P, 2V)
    # Two observed views is the minimum at which "the others" is a real opinion.
    testable = finite3d & (mask.sum(axis=0) >= 2)  # (T, P)

    cams = (
        jnp.asarray(cameras.rvecs),
        jnp.asarray(cameras.tvecs),
        jnp.asarray(cameras.intrs),
        jnp.asarray(cameras.dists),
    )
    kernel = jax.jit(
        jax.vmap(
            _cell_inflation,
            in_axes=(0, 0, 0, 0, 0, None, None, None, None),
        ),
        static_argnums=(6, 7, 8),
    )
    flat = (n_frames * n_points,)
    out = kernel(
        jnp.asarray(x.reshape(*flat, 2 * n_views)),
        jnp.asarray(v.reshape(*flat, 2 * n_views)),
        jnp.asarray(m.reshape(*flat, 2 * n_views)),
        jnp.asarray(safe3d.reshape(*flat, 3)),
        jnp.asarray(testable.reshape(flat)),
        cams,
        float(threshold),
        float(factor),
        int(n_rounds),
    )
    inflated = np.asarray(out).reshape(n_frames, n_points, 2 * n_views)
    # Count views, not dimensions: both of a view's axes inflate together.
    n_inflated = int(np.count_nonzero(inflated[..., ::2] > v[..., ::2] * (1.0 + 1e-9)))
    return unstack_views(inflated, n_views), n_inflated
