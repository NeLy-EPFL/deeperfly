"""The ensemble Kalman smoother end to end: 2D detections in, smoothed 3D out.

This is the assembly of the three pieces the Lightning Pose 3D paper describes:
the ensemble (:mod:`deeperfly.eks.ensemble`), the cross-view variance inflation
(same module), and the nonlinear Kalman filter/smoother whose observation model
is the calibrated camera projection (:mod:`deeperfly.eks.core`).

The 3D estimate it produces is not a triangulation of smoothed 2D -- it is a
*latent* 3D trajectory fitted jointly to every view at once, with the per-frame
observation noise the ensemble and the inflation supply. The 2D it reports is
that trajectory reprojected, which is why a blown detection in one view gets
pulled back onto the animal instead of dragging the 3D point off it.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Bool, Float

from ..rig.cameras import CameraGroup
from .core import fit_smooth_param, smooth_keypoint
from .ensemble import ensemble_statistics, inflate_variances, stack_views

__all__ = ["EksResult", "smooth"]

log = logging.getLogger("deeperfly")

#: Frames averaged for the filter's initial state (the reference uses the same 10).
_INIT_WINDOW = 10

#: Fallback process-noise scale, as a fraction of the track's spatial spread, for a
#: keypoint whose robust lag-1 step comes out zero or undefined (a frozen or barely
#: observed point).
_Q_FALLBACK = 1e-2

#: 1 / Phi^-1(3/4): scales a median absolute deviation to a Gaussian sigma.
_MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class EksResult:
    """What the smoother produced, in deeperfly's array conventions.

    Attributes
    ----------
    pts3d
        The smoothed 3D trajectory ``(T, P, 3)``, NaN for a keypoint no view
        ever saw.
    pts2d
        ``pts3d`` reprojected through the rig ``(V, T, P, 2)``. NaN wherever the
        input was unobserved, unless the caller asked for those cells to be
        filled.
    posterior_var
        The smoother's per-axis posterior variance ``(T, P, 3)``, in the rig's
        world units squared. This is the "uncertainty-aware" output: it grows
        through stretches the views could not pin down and shrinks where they
        agree.
    smooth_param
        The fitted (or supplied) process-noise scale per keypoint ``(P,)``.
        Small means heavily smoothed.
    n_inflated
        How many ``(view, frame, point)`` observations the variance inflation
        down-weighted, and how many were testable at all.
    """

    pts3d: Float[np.ndarray, "T P 3"]
    pts2d: Float[np.ndarray, "V T P 2"]
    posterior_var: Float[np.ndarray, "T P 3"]
    smooth_param: Float[np.ndarray, "P"]
    n_inflated: int = 0
    n_testable: int = 0


def _initial_state(
    init3d: Float[np.ndarray, "T P 3"],
) -> tuple[
    Float[np.ndarray, "P 3"], Float[np.ndarray, "P 3 3"], Float[np.ndarray, "P 3 3"]
]:
    """Per-keypoint prior mean, prior covariance and process-noise *shape*.

    ``Q`` is built from a robust (median-absolute-deviation) lag-1 step of the
    initial track, so it carries the right per-axis proportions and the right
    order of magnitude; the fitted scalar ``s`` then does the actual choosing. A
    keypoint whose track is too sparse or too still to give a step falls back to a
    small fraction of the cloud's spatial spread.
    """
    init3d = np.asarray(init3d, dtype=float)
    with warnings.catch_warnings():  # all-NaN slices are expected here
        warnings.simplefilter("ignore", RuntimeWarning)
        scale = float(np.nanstd(init3d))
        median = np.nanmedian(init3d, axis=0)  # (P, 3)
        head = np.nanmean(init3d[:_INIT_WINDOW], axis=0)  # (P, 3)
        track_var = np.nanvar(init3d, axis=0)  # (P, 3)
        steps = np.diff(init3d, axis=0)  # (T-1, P, 3)
        step_med = np.nanmedian(steps, axis=0)
        mad = np.nanmedian(np.abs(steps - step_med), axis=0)  # (P, 3)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    m0 = np.where(np.isfinite(head), head, median)
    m0 = np.where(np.isfinite(m0), m0, 0.0)

    prior_var = np.where(np.isfinite(track_var), track_var, scale**2)
    prior_var = prior_var + (1e-3 * scale) ** 2  # never exactly singular

    q_var = (_MAD_TO_SIGMA * mad) ** 2
    q_var = np.where(
        np.isfinite(q_var) & (q_var > 0.0), q_var, (_Q_FALLBACK * scale) ** 2
    )

    eye = np.eye(3)
    return m0, prior_var[:, :, None] * eye, q_var[:, :, None] * eye


def _per_keypoint(arr: np.ndarray) -> np.ndarray:
    """``(V, T, P, D) -> (P, T, V * D)``: the axis order the vmapped solver wants."""
    return np.ascontiguousarray(np.swapaxes(stack_views(arr), 0, 1))


def smooth(
    cameras: CameraGroup,
    pts2d: Float[np.ndarray, "V T P 2"] | Float[np.ndarray, "M V T P 2"],
    conf: Float[np.ndarray, "V T P"] | Float[np.ndarray, "M V T P"] | None = None,
    *,
    init3d: Float[np.ndarray, "T P 3"] | None = None,
    smooth_param: float | Float[np.ndarray, "P"] | None = None,
    avg_mode: str = "median",
    var_mode: str = "confidence_weighted_var",
    inflate_vars: bool = True,
    inflate_threshold: float = 5.0,
    inflate_factor: float = 10.0,
    fit_frames: int = 2000,
    fit_iterations: int = 24,
    fill_unobserved: bool = False,
) -> EksResult:
    """Smooth a recording's multi-view 2D into one latent 3D trajectory per keypoint.

    Parameters
    ----------
    cameras
        The calibrated rig. Its projection *is* the observation model, so this
        must be the rig the rest of the run uses (bundle-adjusted when that stage
        is on) -- a stale rig moves the smoother's expectation off the animal.
    pts2d
        2D detections, ``(V, T, P, 2)`` for a single detector or
        ``(M, V, T, P, 2)`` for an ensemble of ``M``. NaN marks unobserved.
    conf
        Matching per-observation confidences, or ``None``.
    init3d
        The 3D estimate used to initialize the filter and to linearize the
        variance inflation, ``(T, P, 3)``. Defaults to a plain DLT triangulation
        of the ensemble consensus; pass the triangulation stage's output to start
        from its outlier handling instead.
    smooth_param
        The process-noise scale: ``None`` (default) fits one per keypoint by
        maximum marginal likelihood, a scalar shares one across all keypoints,
        and an array sets them individually. Smaller smooths more.
    avg_mode, var_mode
        How the ensemble is collapsed (see
        :func:`~deeperfly.eks.ensemble.ensemble_statistics`).
    inflate_vars
        Run the cross-view variance inflation before smoothing. On by default,
        matching the ``eks multicam`` CLI rather than the library signature; it
        is the component that works at ensemble size one, so turning it off on a
        single-detector run leaves little of the method active.
    inflate_threshold, inflate_factor
        The Mahalanobis cutoff and the per-round multiplier (see
        :func:`~deeperfly.eks.ensemble.inflate_variances`).
    fit_frames
        Cap on the leading frames used to *fit* ``smooth_param`` (0 = all).
        Smoothing always runs over every frame; this only bounds the cost of
        estimating one scalar per keypoint, which a few thousand frames settle.
    fit_iterations
        Golden-section iterations per keypoint (see
        :func:`~deeperfly.eks.core.fit_smooth_param`).
    fill_unobserved
        Whether to report the reprojected 2D in cells the detector never
        observed. Off by default: deeperfly treats NaN in ``pts2d`` as "not
        observed", and filling those cells would hand every downstream consumer
        a prediction dressed as a measurement. Turn it on to use the smoother as
        a completion step.

    Returns
    -------
    EksResult
        The smoothed 3D, its reprojection, the posterior variance and the fitted
        smoothing parameters.

    Raises
    ------
    ValueError
        If the array shapes disagree with the rig, or an option is out of range.
    """
    pts2d = np.asarray(pts2d, dtype=float)
    if pts2d.ndim == 4:
        pts2d = pts2d[None]
    if pts2d.ndim != 5 or pts2d.shape[-1] != 2:
        raise ValueError(
            f"pts2d must be (V, T, P, 2) or (M, V, T, P, 2); got {pts2d.shape}"
        )
    n_models, n_views, n_frames, n_points, _ = pts2d.shape
    if n_views != len(cameras):
        raise ValueError(
            f"pts2d has {n_views} views but the rig has {len(cameras)} cameras"
        )
    if n_frames < 2:
        raise ValueError("the smoother needs at least two frames")
    if conf is not None:
        conf = np.asarray(conf, dtype=float)
        if conf.ndim == 3:
            conf = conf[None]
        if conf.shape != pts2d.shape[:4]:
            raise ValueError(f"conf must be {pts2d.shape[:4]}; got {conf.shape}")

    center, var = ensemble_statistics(pts2d, conf, avg_mode=avg_mode, var_mode=var_mode)
    mask: Bool[np.ndarray, "V T P"] = np.isfinite(center).all(axis=-1)
    if init3d is None:
        init3d = cameras.triangulate(np.where(mask[..., None], center, np.nan))
    init3d = np.asarray(init3d, dtype=float)
    if init3d.shape != (n_frames, n_points, 3):
        raise ValueError(
            f"init3d must be ({n_frames}, {n_points}, 3); got {init3d.shape}"
        )

    # A keypoint whose 3D is never determined -- no frame in which two views saw it --
    # has an unobservable latent: the filter would happily report the prior drifting
    # along a single view's ray, which looks like a measurement and is not one. Such a
    # keypoint is dropped to NaN at the end. A keypoint determined in *some* frames and
    # single-view in others is kept: that is the interpolation the smoother is for.
    determined = np.isfinite(init3d).all(axis=-1).any(axis=0)  # (P,)
    if not determined.all():
        # A stricter upstream (RANSAC below its inlier floor) can leave a keypoint
        # un-triangulated that the plain geometry does resolve; fall back to that
        # rather than silently deleting it. Done before the inflation, which skips
        # cells with no finite linearization point.
        fallback = np.asarray(
            cameras.triangulate(np.where(mask[..., None], center, np.nan))
        )
        rescued = np.isfinite(fallback).all(axis=-1).any(axis=0) & ~determined
        if rescued.any():
            init3d = init3d.copy()
            init3d[:, rescued] = fallback[:, rescued]
            determined = determined | rescued
        if not determined.all():
            log.info(
                "eks: %d keypoint(s) are never seen by two views at once and stay NaN",
                int((~determined).sum()),
            )

    n_testable = int(np.count_nonzero(mask & (mask.sum(axis=0) >= 2)[None]))
    n_inflated = 0
    if inflate_vars:
        var, n_inflated = inflate_variances(
            center,
            var,
            mask,
            init3d,
            cameras,
            threshold=inflate_threshold,
            factor=inflate_factor,
        )
        log.info(
            "eks: variance inflation down-weighted %d of %d testable observations "
            "(%.2f%%)",
            n_inflated,
            n_testable,
            100.0 * n_inflated / max(n_testable, 1),
        )

    # A cell the detector never saw carries no information; zero it so no NaN
    # reaches the kernels, and let the mask do the excluding.
    obs = _per_keypoint(np.where(mask[..., None], center, 0.0))  # (P, T, 2V)
    obs_var = _per_keypoint(var)
    obs_mask = _per_keypoint(np.repeat(mask[..., None], 2, axis=-1)).astype(bool)
    m0, prior_cov, q_shape = _initial_state(init3d)
    seen = obs_mask.any(axis=(1, 2)) & determined  # (P,) -- keypoints worth solving

    cams = (
        jnp.asarray(cameras.rvecs),
        jnp.asarray(cameras.tvecs),
        jnp.asarray(cameras.intrs),
        jnp.asarray(cameras.dists),
    )

    if smooth_param is None:
        n_fit = n_frames if not fit_frames else min(int(fit_frames), n_frames)
        log.info(
            "eks: fitting the smoothing parameter for %d keypoint(s) on %d frame(s)",
            int(seen.sum()),
            n_fit,
        )
        fit = jax.jit(
            jax.vmap(
                lambda y, m, v, mu, s0, q: fit_smooth_param(
                    y, m, v, mu, s0, q, cams, n_iter=fit_iterations
                )
            )
        )
        log_s = np.asarray(
            fit(
                jnp.asarray(obs[:, :n_fit]),
                jnp.asarray(obs_mask[:, :n_fit]),
                jnp.asarray(obs_var[:, :n_fit]),
                jnp.asarray(m0),
                jnp.asarray(prior_cov),
                jnp.asarray(q_shape),
            )
        )
        params = np.where(seen, np.exp(log_s), np.nan)
    else:
        params = np.broadcast_to(
            np.asarray(smooth_param, dtype=float), (n_points,)
        ).astype(float)
        if not np.all(np.isfinite(params) & (params > 0)):
            raise ValueError("smooth_param must be positive and finite")
        # Report NaN for the keypoints that were not solved, as the fitted branch does:
        # a number here would imply an answer that the arrays do not contain.
        params = np.where(seen, params, np.nan)

    log.info("eks: smoothing %d frames x %d keypoints", n_frames, n_points)
    run = jax.jit(
        jax.vmap(
            lambda y, m, v, mu, s0, q, s: smooth_keypoint(y, m, v, mu, s0, q, s, cams)
        )
    )
    means, covs = run(
        jnp.asarray(obs),
        jnp.asarray(obs_mask),
        jnp.asarray(obs_var),
        jnp.asarray(m0),
        jnp.asarray(prior_cov),
        jnp.asarray(q_shape),
        jnp.asarray(np.where(seen, params, 1.0)),
    )
    # np.asarray of a JAX array is read-only; copy before masking the unseen columns.
    pts3d = np.array(np.swapaxes(np.asarray(means), 0, 1))  # (T, P, 3)
    posterior_var = np.array(
        np.swapaxes(np.diagonal(np.asarray(covs), axis1=-2, axis2=-1), 0, 1)
    )  # (T, P, 3)
    unseen = ~seen
    pts3d[:, unseen] = np.nan
    posterior_var[:, unseen] = np.nan

    reprojected = np.asarray(cameras.project(pts3d))  # (V, T, P, 2)
    if not fill_unobserved:
        reprojected = np.where(mask[..., None], reprojected, np.nan)

    return EksResult(
        pts3d=pts3d,
        pts2d=reprojected,
        posterior_var=posterior_var,
        smooth_param=params,
        n_inflated=n_inflated,
        n_testable=n_testable,
    )
