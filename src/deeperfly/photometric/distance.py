"""Truncated distance transform of a leg-response map (the chamfer potential).

The reprojected leg bones are pulled onto the image leg pixels by minimizing the
distance from each sampled point on a bone to the nearest leg pixel. That distance
field is precomputed once per (view, frame) as a Euclidean distance transform of the
thresholded :mod:`~deeperfly.photometric.leg_response` map, then *truncated* at
``trunc_px``.

Truncation does two things: it bounds the convergence-basin radius (a badly-off line
is pulled toward a *nearby* leg, not an arbitrary structure across the image), and it
turns the region beyond ``trunc_px`` into a flat plateau with zero gradient -- so an
occluded or absent far leg (whose bone reprojects onto empty background) saturates at
``trunc_px`` and is ignored by the robust loss rather than mis-pulled.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
from jaxtyping import Float


def truncated_dt(
    response: Float[np.ndarray, "H W"],
    *,
    threshold: float = 0.15,
    trunc_px: float = 20.0,
) -> Float[np.ndarray, "H W"]:
    """Euclidean distance (px) to the nearest leg pixel, clipped to ``trunc_px``.

    Parameters
    ----------
    response
        Non-negative leg-response map (see :func:`deeperfly.photometric.leg_response`).
    threshold
        Response value above which a pixel counts as a leg pixel.
    trunc_px
        Distances are clipped to this many pixels. Choose ~2-3 leg widths: large
        enough to cover the reprojection error, small enough that a bone cannot be
        pulled across the whole frame.

    Returns
    -------
    The truncated distance field, ``float32``. Zero on leg pixels, growing to
    ``trunc_px`` away from them; a response with no leg pixels yields a constant
    ``trunc_px`` field (that view/frame contributes no gradient).
    """
    mask = response >= threshold
    if mask.any():
        dist = ndi.distance_transform_edt(~mask)
    else:
        dist = np.full(response.shape, trunc_px, dtype=np.float64)
    return np.minimum(dist, trunc_px).astype(np.float32)
