"""Gaussian heatmap targets, the sub-pixel decode, and the masked training loss.

Three numerics that have to stay **stable** across the whole label -> train -> predict ->
correct loop, because a change in any of them silently shifts every pixel error reported
downstream:

**Targets** are peak-1.0 separable Gaussians rendered at the *fractional* heatmap
coordinate. A label at ``x = 40.3`` trains a peak at 40.3, not at cell 40. Rounding here
bakes in a half-cell bias -- ``STRIDE / 2`` input pixels -- which is exactly the class of bug
already found and fixed once in this package's own decode
(:mod:`deeperfly.pose2d.inference`), and it is invisible in the loss curve.

**Decode** is the DARK/parabolic refinement, never the raw argmax. The argmax quantizes to
whole heatmap cells, i.e. ``STRIDE`` input pixels; a quadratic fit to the peak's immediate
neighbours recovers most of that. Training metrics, evaluation and the editor must all use
*this* function, or they disagree by a constant that nobody attributes correctly.

**Loss** is foreground-weighted MSE under a *continuous* per-label weight, normalized by the
weight mass actually present. Real labels are sparse -- a frame labels the near side, or six
of 38 points -- so a zero-weight slot must contribute to neither numerator nor denominator.
Otherwise a sparsely-labeled sample dilutes its batch in proportion to how little of it was
labeled, which is the opposite of what should happen.

The numerics deliberately match the ones the ``dfpose`` research trainer settled on, so a
tuned pipeline can move onto this core without changing a single reported number.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "STRIDE",
    "masked_heatmap_loss",
    "refined_argmax",
    "render_gaussian_targets",
]

#: Heatmap-to-input downsampling. A decode that quantizes to a cell is therefore wrong by up
#: to this many input pixels, which is why :func:`refined_argmax` exists.
STRIDE = 4


def render_gaussian_targets(
    keypoints_hm, visible, *, hw: tuple[int, int], sigma: float = 2.0
):
    """Peak-1.0 Gaussian targets from keypoints in **heatmap** pixels.

    Parameters
    ----------
    keypoints_hm
        ``(P, 2)`` or ``(B, P, 2)`` positions in heatmap coordinates -- fractional, and used
        fractionally.
    visible
        ``(P,)`` or ``(B, P)``. A zero leaves that channel all-zero rather than rendering a
        peak somewhere arbitrary.
    hw
        Heatmap ``(height, width)``.
    sigma
        Gaussian sigma, in heatmap pixels.

    Returns
    -------
    np.ndarray
        ``(P, H, W)`` or ``(B, P, H, W)`` float32.
    """
    pts = np.asarray(keypoints_hm, dtype=np.float32)
    vis = np.asarray(visible, dtype=np.float32)
    batched = pts.ndim == 3
    if not batched:
        pts, vis = pts[None], vis[None]
    height, width = int(hw[0]), int(hw[1])

    ys = np.arange(height, dtype=np.float32)
    xs = np.arange(width, dtype=np.float32)
    # Separable: exp(-(dy^2 + dx^2)/2s^2) factorizes, so this is two 1-D exponentials and an
    # outer product rather than a (P, H, W) distance field.
    dy = ys[None, None, :] - pts[..., 1][..., None]  # (B, P, H)
    dx = xs[None, None, :] - pts[..., 0][..., None]  # (B, P, W)
    denom = 2.0 * float(sigma) ** 2
    with np.errstate(invalid="ignore"):
        gy = np.exp(-(dy**2) / denom)
        gx = np.exp(-(dx**2) / denom)
    out = gy[..., :, None] * gx[..., None, :]  # (B, P, H, W)
    # An invisible or non-finite keypoint gets an all-zero channel: a NaN coordinate would
    # otherwise propagate through the exponential and poison the whole target.
    ok = (vis > 0) & np.isfinite(pts).all(axis=-1)
    out = np.where(ok[..., None, None], out, 0.0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32) if batched else out[0].astype(np.float32)


def refined_argmax(heatmaps):
    """Sub-pixel peak per channel, in **heatmap** pixels, by parabolic refinement.

    The raw argmax quantizes to a cell, i.e. :data:`STRIDE` input pixels. A quadratic fit to
    the peak's two immediate neighbours in each axis recovers most of that -- the DARK
    refinement -- and every consumer must use it, or their numbers differ by a constant that
    looks like a model regression.

    Parameters
    ----------
    heatmaps
        ``(P, H, W)`` or ``(B, P, H, W)``.

    Returns
    -------
    xy, score : np.ndarray
        ``(..., P, 2)`` refined peaks and ``(..., P)`` peak values.
    """
    hm = np.asarray(heatmaps, dtype=np.float32)
    batched = hm.ndim == 4
    if not batched:
        hm = hm[None]
    _, _, height, width = hm.shape
    flat = hm.reshape(hm.shape[0], hm.shape[1], -1)
    idx = flat.argmax(axis=-1)
    score = np.take_along_axis(flat, idx[..., None], axis=-1)[..., 0]
    py, px = np.divmod(idx, width)
    xy = np.stack([px.astype(np.float32), py.astype(np.float32)], axis=-1)

    for axis, (coord, limit) in enumerate(((px, width), (py, height))):
        # The parabola through (c-1, c, c+1). Skipped at a border, where one neighbour does
        # not exist -- a peak on the edge is already suspect, and inventing a neighbour would
        # bias it inward.
        inner = (coord > 0) & (coord < limit - 1)
        if not inner.any():
            continue
        bi, pi = np.nonzero(inner)
        ci = coord[bi, pi]
        if axis == 0:
            left, mid, right = (
                hm[bi, pi, py[bi, pi], ci - 1],
                hm[bi, pi, py[bi, pi], ci],
                hm[bi, pi, py[bi, pi], ci + 1],
            )
        else:
            left, mid, right = (
                hm[bi, pi, ci - 1, px[bi, pi]],
                hm[bi, pi, ci, px[bi, pi]],
                hm[bi, pi, ci + 1, px[bi, pi]],
            )
        curve = left - 2.0 * mid + right
        with np.errstate(invalid="ignore", divide="ignore"):
            offset = np.where(np.abs(curve) > 1e-9, (left - right) / (2.0 * curve), 0.0)
        # |offset| > 0.5 means the fit disagrees with the argmax about which cell holds the
        # peak, which happens on a flat or bimodal channel. Clamped rather than trusted.
        xy[bi, pi, axis] += np.clip(np.nan_to_num(offset), -0.5, 0.5)
    return (xy, score) if batched else (xy[0], score[0])


def masked_heatmap_loss(pred, target, weight, *, foreground: float = 10.0):
    """Foreground-weighted MSE, normalized by the weight mass actually present.

    Parameters
    ----------
    pred, target
        ``(B, P, H, W)`` predicted and target heatmaps (torch tensors or arrays).
    weight
        ``(B, P)`` per-label weight. Zero means "this slot carries no label", and such a slot
        contributes to **neither** numerator nor denominator -- so a frame with six of 38
        points labeled is not penalized for the 32 it never claimed.
    foreground
        Extra weight on target-positive pixels. A heatmap is ~99% background, so unweighted
        MSE is minimized by predicting zero everywhere.

    Returns
    -------
    The scalar loss, in whichever framework the inputs came from.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover -- torch is a core dependency
        torch = None

    if torch is not None and isinstance(pred, torch.Tensor):
        w = weight.to(pred.dtype)[..., None, None]
        fg = 1.0 + (foreground - 1.0) * (target > 0.01).to(pred.dtype)
        mass = (fg * w).sum()
        return ((pred - target) ** 2 * fg * w).sum() / mass.clamp_min(1e-8)

    pred_a = np.asarray(pred, dtype=np.float64)
    target_a = np.asarray(target, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)[..., None, None]
    fg = 1.0 + (foreground - 1.0) * (target_a > 0.01)
    mass = float((fg * w).sum())
    return float(((pred_a - target_a) ** 2 * fg * w).sum() / max(mass, 1e-8))
