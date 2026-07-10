"""Turn raw frames into a non-negative "legness" map that fires on legs, not the body.

The fly body and the legs are *both* bright against the dark background, so a plain
intensity/foreground objective would pull a reprojected bone onto the body just as
readily as onto a leg. The response here keys on *morphology* instead: a multiscale
Hessian **ridge** filter tuned to the leg width responds to thin elongated structures
(legs) and rejects the wide body blob by construction. It is optionally gated by a
temporal-foreground mask (median-background subtraction) to suppress static clutter,
with a soft floor so a momentarily still leg does not vanish.

:func:`paint_out_bones` erases the accurately-reprojected *near* legs from the map so a
far bone can only match *unexplained* (far-leg) response -- the fix for the
wrong-correspondence trap where a far leg reprojects into the near-leg cluster.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.ndimage as ndi
from jaxtyping import Float, UInt8

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

Polarity = Literal["bright", "dark", "auto"]


def to_gray(frames: UInt8[np.ndarray, "F H W 3"]) -> Float[np.ndarray, "F H W"]:
    """RGB uint8 frames -> float32 luminance in ``[0, 1]``."""
    return (frames.astype(np.float32) / 255.0) @ _LUMA


def ridge_response(
    gray: Float[np.ndarray, "F H W"],
    *,
    leg_width_px: float = 5.0,
    scales: tuple[float, ...] = (0.7, 1.0, 1.5),
    polarity: Polarity = "bright",
) -> Float[np.ndarray, "F H W"]:
    """Multiscale Hessian ridge filter (thin-line detector), normalized to ``[0, 1]``.

    For each scale ``sigma = leg_width_px/2 * s`` the Hessian eigenvalue of largest
    magnitude ``lam1`` is computed; a bright line has ``lam1`` strongly negative (a dark
    line, positive). The scale-normalized response ``sigma**2 * relu(-/+ lam1)`` is
    maxed over scales so both thick near legs and thin far legs light up.
    """
    out = np.zeros_like(gray)
    for f in range(gray.shape[0]):
        g = gray[f]
        best = np.zeros_like(g)
        for s in scales:
            sigma = max(0.6, leg_width_px / 2.0 * s)
            gxx = ndi.gaussian_filter(g, sigma, order=(0, 2))
            gyy = ndi.gaussian_filter(g, sigma, order=(2, 0))
            gxy = ndi.gaussian_filter(g, sigma, order=(1, 1))
            root = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4.0 * gxy**2, 0.0))
            lam_a = 0.5 * ((gxx + gyy) + root)
            lam_b = 0.5 * ((gxx + gyy) - root)
            lam1 = np.where(np.abs(lam_a) >= np.abs(lam_b), lam_a, lam_b)
            if polarity == "bright":
                r = np.maximum(-lam1, 0.0)
            elif polarity == "dark":
                r = np.maximum(lam1, 0.0)
            else:  # "auto" handled by caller; treat as magnitude here
                r = np.abs(lam1)
            best = np.maximum(best, sigma**2 * r)
        out[f] = best
    hi = np.percentile(out, 99.5)
    return np.clip(out / hi, 0.0, 1.0) if hi > 0 else out


def _foreground_gate(
    gray: Float[np.ndarray, "F H W"],
    *,
    fg_dilate_px: float = 6.0,
    fg_floor: float = 0.3,
    fg_thresh: float = 0.15,
) -> Float[np.ndarray, "F H W"]:
    """Soft temporal-foreground gate from median-background subtraction."""
    bg = np.median(gray, axis=0)
    fg = np.abs(gray - bg[None])
    fgn = np.clip(fg / max(np.percentile(fg, 99.5), 1e-6), 0.0, 1.0)
    gate = np.empty_like(fgn)
    for f in range(fgn.shape[0]):
        m = ndi.binary_dilation(
            fgn[f] > fg_thresh, iterations=int(max(1, fg_dilate_px))
        )
        gate[f] = np.where(m, fg_floor + (1.0 - fg_floor) * fgn[f], 0.0)
    return gate


def leg_response(
    frames: UInt8[np.ndarray, "F H W 3"],
    *,
    method: str = "ridge_fg",
    leg_width_px: float = 5.0,
    polarity: Polarity = "auto",
    fg_dilate_px: float = 6.0,
    fg_floor: float = 0.3,
) -> Float[np.ndarray, "F H W"]:
    """Legness map in ``[0, 1]`` -- ridge filter, optionally foreground-gated.

    ``method``: ``"ridge_fg"`` (default) = ridge * temporal-foreground gate;
    ``"ridge"`` = ridge only; ``"fg"`` = foreground only.
    ``polarity="auto"`` picks bright/dark by whichever gives more response inside the
    foreground.
    """
    gray = to_gray(frames)
    if polarity == "auto":
        gate = _foreground_gate(gray, fg_dilate_px=fg_dilate_px, fg_floor=fg_floor)
        m = gate > 0
        rb = ridge_response(gray, leg_width_px=leg_width_px, polarity="bright")
        rd = ridge_response(gray, leg_width_px=leg_width_px, polarity="dark")
        # Compare the *peak* (high-percentile) ridge, not the mean: legs are a small
        # fraction of the foreground, so the mean is swamped by body/gap texture and
        # can pick the wrong sign. The correct polarity has the stronger leg ridges.
        sb = np.percentile(rb[m], 99) if m.any() else 0.0
        sd = np.percentile(rd[m], 99) if m.any() else 0.0
        pol: Polarity = "bright" if sb >= sd else "dark"
    else:
        pol = polarity

    if method == "fg":
        return _foreground_gate(gray, fg_dilate_px=fg_dilate_px, fg_floor=fg_floor)
    ridge = ridge_response(gray, leg_width_px=leg_width_px, polarity=pol)
    if method == "ridge":
        return ridge
    gate = _foreground_gate(gray, fg_dilate_px=fg_dilate_px, fg_floor=fg_floor)
    return ridge * gate


def paint_out_bones(
    response: Float[np.ndarray, "H W"],
    segments_xy: np.ndarray,
    *,
    width_px: float,
) -> Float[np.ndarray, "H W"]:
    """Zero the response along a set of 2D line segments (erase near legs).

    ``segments_xy`` is ``(B, 2, 2)`` = per bone the two endpoint ``(x, y)`` pixels in the
    response frame. A tube of ``width_px`` around each is set to zero, so a far bone
    cannot match response explained by an already-well-fit near leg.
    """
    import cv2

    out = response.copy()
    thick = int(round(max(1.0, width_px)))
    for pa, pb in segments_xy:
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(
                out,
                (int(pa[0]), int(pa[1])),
                (int(pb[0]), int(pb[1])),
                0.0,
                thick,
                cv2.LINE_AA,
            )
    return out
