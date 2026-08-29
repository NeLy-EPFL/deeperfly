"""The nonlinear Kalman filter and smoother the EKS runs on, in JAX.

One keypoint's latent state is its 3D world position ``z_t``; the observation is
that point seen in every calibrated view at once. The model is

.. math::

    z_t \\sim \\mathcal{N}(z_{t-1},\\; s\\,Q) \\qquad
    x_t \\sim \\mathcal{N}(h(z_t),\\; R_t)

with ``h`` the *full* camera projection of :func:`deeperfly.geometry.project_full_one`
stacked over the ``V`` views (so ``x_t`` is a ``2V``-vector of pixels), ``R_t`` the
diagonal per-observation variance, ``Q`` a per-axis process-noise shape and ``s`` a
single scalar per keypoint fitted by maximum likelihood.

Because ``h`` is nonlinear the filter is an *extended* Kalman filter -- it linearizes
``h`` about the predicted mean with :func:`jax.jacfwd` -- while the dynamics stay
linear (a random walk, ``A = I``), so the backward pass is the exact Rauch-Tung-Striebel
recursion.

Missing observations
--------------------
deeperfly's 2D array is NaN wherever a ``(view, point)`` pair is unobserved -- no
pathway maps it, the operator declared the limb absent, or RANSAC dropped the view.
Those cells are handled *exactly*, not imputed: a masked dimension gets a zeroed
Jacobian row, a zeroed residual and unit variance, which makes the innovation
covariance block-diagonal with a 1 in that slot. Its Kalman-gain column is then
identically zero and it contributes nothing to the log-likelihood, so the recursion
is precisely the one for the observed dimensions alone. A keypoint seen by fewer
than two views in some stretch is therefore not dropped (as triangulation must drop
it) -- the filter simply coasts on the dynamics and the smoother interpolates it
from both sides.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Float

from ..geometry import project_full_one

__all__ = [
    "project_all",
    "ekf_filter",
    "rts_smooth",
    "smooth_keypoint",
    "fit_smooth_param",
]

_LOG_2PI = math.log(2.0 * math.pi)

#: Golden-ratio conjugate, the shrink factor of :func:`fit_smooth_param`'s bracket.
_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0

#: Floor on an observation variance, in px^2. A confidence of exactly 1.0 would
#: otherwise assert a pixel-perfect detection and let one frame dominate the fit.
_VAR_FLOOR = 1e-6


def project_all(
    pt3d: Float[Array, "3"],
    rvecs: Float[Array, "V 3"],
    tvecs: Float[Array, "V 3"],
    intrs: Float[Array, "V 4"],
    dists: Float[Array, "V K"],
) -> Float[Array, "2V"]:
    """``h(z)``: one 3D point projected into every camera, flattened.

    The observation function of the nonlinear EKS. It is the same projection the
    rest of deeperfly reprojects with (distortion included), so the smoother's
    geometry and the reprojection overlay's geometry cannot drift apart.

    Parameters
    ----------
    pt3d
        A world point of shape ``(3,)``.
    rvecs, tvecs, intrs, dists
        The rig's stacked camera parameters (see
        :class:`~deeperfly.rig.cameras.CameraGroup`).

    Returns
    -------
    Float[Array, "2V"]
        ``[u_0, v_0, u_1, v_1, ...]`` in pixels, view-major.
    """
    xy = jax.vmap(project_full_one, in_axes=(None, 0, 0, 0, 0))(
        pt3d, rvecs, tvecs, intrs, dists
    )
    return xy.reshape(-1)


def _sym(matrix: Float[Array, "n n"]) -> Float[Array, "n n"]:
    """Re-symmetrize a covariance that round-off has nudged off-symmetric."""
    return 0.5 * (matrix + matrix.T)


def _update(
    m_pred: Float[Array, "3"],
    p_pred: Float[Array, "3 3"],
    y: Float[Array, "2V"],
    mask: Bool[Array, "2V"],
    var: Float[Array, "2V"],
    cams: tuple,
) -> tuple[Float[Array, "3"], Float[Array, "3 3"], Float[Array, ""]]:
    """One EKF measurement update, returning ``(mean, covariance, -log p(y))``.

    The covariance uses the Joseph form, which stays positive-definite under
    round-off where the textbook ``(I - KH) P`` does not -- and this recursion runs
    for tens of thousands of steps per keypoint.
    """

    def h(z):
        return project_all(z, *cams)

    y_hat = h(m_pred)
    # Zeroing the Jacobian rows of the unobserved dimensions is what makes them
    # drop out exactly: their Kalman-gain columns become zero regardless of R.
    jac = jax.jacfwd(h)(m_pred) * mask[:, None]  # (2V, 3)
    resid = jnp.where(mask, y - y_hat, 0.0)
    # Unit variance on a masked dimension contributes log(1) = 0 to the log-determinant,
    # so the likelihood below is the marginal of the observed dimensions alone.
    obs_var = jnp.where(mask, jnp.maximum(var, _VAR_FLOOR), 1.0)

    innov = _sym(jac @ p_pred @ jac.T) + jnp.diag(obs_var)
    chol = jnp.linalg.cholesky(innov)
    gain = jax.scipy.linalg.cho_solve((chol, True), jac @ p_pred).T  # (3, 2V)
    mean = m_pred + gain @ resid
    joseph = jnp.eye(3) - gain @ jac
    cov = _sym(joseph @ p_pred @ joseph.T + (gain * obs_var) @ gain.T)

    whitened = jax.scipy.linalg.cho_solve((chol, True), resid)
    n_obs = jnp.sum(mask)
    nll = 0.5 * (
        resid @ whitened + 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol))) + n_obs * _LOG_2PI
    )
    return mean, cov, nll


def ekf_filter(
    y: Float[Array, "T 2V"],
    mask: Bool[Array, "T 2V"],
    var: Float[Array, "T 2V"],
    m0: Float[Array, "3"],
    s0: Float[Array, "3 3"],
    q: Float[Array, "3 3"],
    cams: tuple,
) -> tuple[
    Float[Array, "T 3"],
    Float[Array, "T 3 3"],
    Float[Array, "T 3"],
    Float[Array, "T 3 3"],
    Float[Array, ""],
]:
    """Run the extended Kalman filter forward over one keypoint's track.

    Parameters
    ----------
    y, mask, var
        The stacked multi-view observations, their observed-ness, and their
        per-dimension variance, each over ``T`` frames and ``2V`` dimensions.
    m0, s0
        The prior mean and covariance at ``t = 0`` (which is the ``t = 0``
        prediction: no dynamics step precedes the first observation).
    q
        The process-noise covariance, already scaled by the smoothing parameter.
    cams
        ``(rvecs, tvecs, intrs, dists)`` for :func:`project_all`.

    Returns
    -------
    m_pred, p_pred : Float[Array, "T 3"], Float[Array, "T 3 3"]
        The one-step-ahead predictions (the smoother needs them).
    m_filt, p_filt : Float[Array, "T 3"], Float[Array, "T 3 3"]
        The filtered posteriors.
    nll : Float[Array, ""]
        The negative marginal log-likelihood of the whole track -- the objective
        :func:`fit_smooth_param` minimizes.
    """

    def step(carry, xs):
        m_prior, p_prior = carry
        y_t, mask_t, var_t = xs
        m_post, p_post, nll_t = _update(m_prior, p_prior, y_t, mask_t, var_t, cams)
        # Random-walk dynamics (A = I): tomorrow's prior is today's posterior, blurred.
        return (m_post, p_post + q), (m_prior, p_prior, m_post, p_post, nll_t)

    _, out = lax.scan(step, (m0, s0), (y, mask, var))
    m_pred, p_pred, m_filt, p_filt, nll = out
    return m_pred, p_pred, m_filt, p_filt, jnp.sum(nll)


def rts_smooth(
    m_pred: Float[Array, "T 3"],
    p_pred: Float[Array, "T 3 3"],
    m_filt: Float[Array, "T 3"],
    p_filt: Float[Array, "T 3 3"],
) -> tuple[Float[Array, "T 3"], Float[Array, "T 3 3"]]:
    """The backward Rauch-Tung-Striebel pass over a filtered track.

    Exact rather than approximate: the *dynamics* are linear (``A = I``), and only
    the observation model was linearized, so no second approximation enters here.
    """

    def step(carry, xs):
        m_next, p_next = carry
        m_f, p_f, m_p, p_p = xs
        # gain = P_f A^T (P_pred)^-1 with A = I; both operands are symmetric, so
        # solve(P_pred, P_f).T is P_f @ inv(P_pred) without forming the inverse.
        gain = jnp.linalg.solve(p_p, p_f).T
        mean = m_f + gain @ (m_next - m_p)
        cov = _sym(p_f + gain @ (p_next - p_p) @ gain.T)
        return (mean, cov), (mean, cov)

    _, out = lax.scan(
        step,
        (m_filt[-1], p_filt[-1]),
        (m_filt[:-1], p_filt[:-1], m_pred[1:], p_pred[1:]),
        reverse=True,
    )
    means, covs = out
    return (
        jnp.concatenate([means, m_filt[-1][None]], axis=0),
        jnp.concatenate([covs, p_filt[-1][None]], axis=0),
    )


def fit_smooth_param(
    y: Float[Array, "T 2V"],
    mask: Bool[Array, "T 2V"],
    var: Float[Array, "T 2V"],
    m0: Float[Array, "3"],
    s0: Float[Array, "3 3"],
    q: Float[Array, "3 3"],
    cams: tuple,
    *,
    bounds: tuple[float, float] = (-8.0, 8.0),
    n_iter: int = 24,
) -> Float[Array, ""]:
    """Fit one keypoint's smoothing parameter by maximum marginal likelihood.

    Returns ``log(s)``, where ``s`` scales the process noise: small ``s`` trusts the
    dynamics and smooths hard, large ``s`` trusts the detector and follows it. The
    objective is the EKF's own negative marginal log-likelihood, so nothing is
    held out -- the likelihood already penalizes both over- and under-smoothing,
    which is why the paper reports a single well-defined optimum per keypoint.

    The search is a golden-section bracket on ``log(s)``, not the reference
    implementation's Adam: the problem is one-dimensional and unimodal, so a
    bracket needs no learning rate, no convergence tolerance and no restart
    heuristics, and it costs one likelihood evaluation per iteration. The
    objective, the parameterization and the ``(-8, 8)`` bounds are the paper's.

    Parameters
    ----------
    y, mask, var, m0, s0, q, cams
        As for :func:`ekf_filter`; ``q`` is the *unscaled* process-noise shape.
    bounds
        The ``log(s)`` bracket to search.
    n_iter
        Golden-section iterations. Each shrinks the bracket by ~0.618, so the
        default narrows a 16-wide log range to under 1e-4.

    Returns
    -------
    Float[Array, ""]
        The fitted ``log(s)``.
    """

    def loss(log_s):
        nll = ekf_filter(y, mask, var, m0, s0, jnp.exp(log_s) * q, cams)[4]
        return jnp.where(jnp.isfinite(nll), nll, jnp.inf)

    lo, hi = float(bounds[0]), float(bounds[1])
    lo = jnp.asarray(lo, dtype=float)
    hi = jnp.asarray(hi, dtype=float)
    left = hi - _INV_PHI * (hi - lo)
    right = lo + _INV_PHI * (hi - lo)

    def body(_, state):
        lo, hi, left, right, f_left, f_right = state
        keep_left = f_left < f_right
        new_lo = jnp.where(keep_left, lo, left)
        new_hi = jnp.where(keep_left, right, hi)
        span = _INV_PHI * (new_hi - new_lo)
        probe = jnp.where(keep_left, new_hi - span, new_lo + span)
        f_probe = loss(probe)
        return (
            new_lo,
            new_hi,
            jnp.where(keep_left, probe, right),
            jnp.where(keep_left, left, probe),
            jnp.where(keep_left, f_probe, f_right),
            jnp.where(keep_left, f_left, f_probe),
        )

    state = (lo, hi, left, right, loss(left), loss(right))
    _, _, left, right, f_left, f_right = lax.fori_loop(0, n_iter, body, state)
    return jnp.where(f_left < f_right, left, right)


def smooth_keypoint(
    y: Float[Array, "T 2V"],
    mask: Bool[Array, "T 2V"],
    var: Float[Array, "T 2V"],
    m0: Float[Array, "3"],
    s0: Float[Array, "3 3"],
    q: Float[Array, "3 3"],
    smooth_param: Float[Array, ""],
    cams: tuple,
) -> tuple[Float[Array, "T 3"], Float[Array, "T 3 3"]]:
    """Filter then smooth one keypoint at a given ``s``, returning the posterior."""
    m_pred, p_pred, m_filt, p_filt, _ = ekf_filter(
        y, mask, var, m0, s0, smooth_param * q, cams
    )
    return rts_smooth(m_pred, p_pred, m_filt, p_filt)
