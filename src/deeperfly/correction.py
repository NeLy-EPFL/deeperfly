"""3D error correction: outliers and temporal smoothing.

- **Outlier handling** -- flag observations whose reprojection error exceeds a
  pixel threshold (:func:`flag_outliers`) and drop them to ``NaN``
  (:func:`drop_outliers`) so the point can be re-triangulated from the rest.
- **Temporal smoothing** -- NaN-aware Gaussian (:func:`smooth_gaussian`) or a
  streaming 1-Euro filter (:class:`OneEuroFilter` / :func:`smooth_one_euro`).

Robust loss (Huber) and bone-length priors live in bundle adjustment, not here.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Bool, Float
from scipy.ndimage import gaussian_filter1d

# -- outliers ----------------------------------------------------------------


def flag_outliers(
    reproj_error: Float[np.ndarray, "V *pts"],
    threshold: float = 40.0,
) -> Bool[np.ndarray, "V *pts"]:
    """Boolean mask of observations whose reprojection error exceeds ``threshold``.

    ``NaN`` errors (unobserved entries) compare ``False`` and are never flagged.
    """
    return np.asarray(reproj_error) > threshold


def drop_outliers(
    pts2d: Float[np.ndarray, "V *pts 2"],
    mask: Bool[np.ndarray, "V *pts"],
) -> Float[np.ndarray, "V *pts 2"]:
    """Return a copy of ``pts2d`` with flagged (view, point) entries set to NaN."""
    out = np.array(pts2d, dtype=float)
    out[np.asarray(mask)] = np.nan
    return out


# -- temporal smoothing ------------------------------------------------------


def smooth_gaussian(
    pts_seq: Float[np.ndarray, "T N 3"],
    sigma: float,
) -> Float[np.ndarray, "T N 3"]:
    """NaN-aware Gaussian smoothing along time (axis 0) via normalized convolution.

    Missing samples (``NaN``) are excluded from the weighted average rather than
    poisoning their neighbors; positions all-NaN within a window stay ``NaN``.
    """
    a = np.asarray(pts_seq, dtype=float)
    nan = np.isnan(a)
    filled = np.where(nan, 0.0, a)
    num = gaussian_filter1d(filled, sigma, axis=0, mode="nearest")
    weight = gaussian_filter1d((~nan).astype(float), sigma, axis=0, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / np.where(weight == 0, np.nan, weight)


class _LowPass:
    """Exponential low-pass filter holding the previous smoothed value."""

    def __init__(self) -> None:
        self.prev: np.ndarray | None = None

    def __call__(self, value: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        s = value if self.prev is None else alpha * value + (1 - alpha) * self.prev
        self.prev = s
        return s


class OneEuroFilter:
    """Streaming 1-Euro filter (Casiez et al. 2012) for vector-valued signals.

    Low cutoff for slow motion (low jitter), higher cutoff as speed rises (low
    lag), controlled by ``mincutoff`` and ``beta``. Filters each array element
    independently.
    """

    def __init__(
        self,
        freq: float,
        *,
        mincutoff: float = 1.0,
        beta: float = 0.0,
        dcutoff: float = 1.0,
    ) -> None:
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._prev: np.ndarray | None = None

    def _alpha(self, cutoff: float | np.ndarray) -> np.ndarray:
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dx = np.zeros_like(x) if self._prev is None else (x - self._prev) * self.freq
        self._prev = x
        edx = self._dx(dx, self._alpha(self.dcutoff))
        cutoff = self.mincutoff + self.beta * np.abs(edx)
        return self._x(x, self._alpha(cutoff))


def smooth_one_euro(
    pts_seq: Float[np.ndarray, "T N 3"],
    fps: float,
    *,
    mincutoff: float = 1.0,
    beta: float = 0.0,
    dcutoff: float = 1.0,
) -> Float[np.ndarray, "T N 3"]:
    """Apply a :class:`OneEuroFilter` frame-by-frame over time (axis 0).

    NaN samples are held at the previous filtered value (the filter state is not
    updated), so gaps neither propagate NaN nor inject spurious motion.
    """
    a = np.asarray(pts_seq, dtype=float)
    filt = OneEuroFilter(fps, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)
    out = np.empty_like(a)
    last = np.full(a.shape[1:], np.nan)
    for t in range(a.shape[0]):
        frame = a[t]
        valid = np.isfinite(frame)
        filtered = filt(np.where(valid, frame, np.nan_to_num(last)))
        last = np.where(valid, filtered, last)
        out[t] = np.where(valid, filtered, np.nan)
    return out
