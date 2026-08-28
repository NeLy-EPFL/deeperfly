"""Per-point RGB from a skeleton's colors, with no plotting deps.

A skeleton carries one hex color per point and one per edge. The OpenCV overlay and
compositor
(:mod:`deeperfly.visualization.opencv`, :mod:`deeperfly.visualization.compose`) draw
straight into image arrays, so the conversion to RGB lives here as plain NumPy with no
matplotlib dependency.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..skeleton import Skeleton


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    """Parse a ``#rgb`` / ``#rrggbb`` hex color to RGB floats in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``value`` is not a 3- or 6-digit hex color.
    """
    h = value.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"expected a #rgb or #rrggbb hex color, got {value!r}")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b)


def point_colors_rgb(
    skeleton: Skeleton, colors: Sequence[str] | None = None
) -> np.ndarray:
    """``(P, 3)`` RGB floats in ``[0, 1]``, one per tracked point.

    Parameters
    ----------
    skeleton
        Skeleton supplying :attr:`~deeperfly.skeleton.Skeleton.point_colors`.
    colors
        Optional per-point hex override, in point order.

    Returns
    -------
    np.ndarray
        ``(P, 3)`` RGB floats in ``[0, 1]``.
    """
    hexes = skeleton.point_colors if colors is None else tuple(colors)
    if len(hexes) != skeleton.n_points:
        raise ValueError(
            f"{len(hexes)} colors for a {skeleton.n_points}-point skeleton"
        )
    return np.asarray([hex_to_rgb(h) for h in hexes], dtype=float).reshape(-1, 3)


def edge_colors_rgb(
    skeleton: Skeleton, colors: Sequence[str] | None = None
) -> np.ndarray:
    """``(E, 3)`` RGB floats in ``[0, 1]``, one per edge.

    ``colors`` is the same per-POINT hex override :func:`point_colors_rgb` takes. Given
    one, the edges are re-derived from it by the endpoint average rather than read off
    the skeleton -- an override that recolors the joints has to recolor what joins them,
    or a drawing in someone else's palette keeps the skeleton's own edges.
    """
    if colors is None:
        return np.asarray(
            [hex_to_rgb(h) for h in skeleton.edge_colors], dtype=float
        ).reshape(-1, 3)
    pts = point_colors_rgb(skeleton, colors)
    rows = np.asarray(skeleton.edges).reshape(-1, 2)
    if not len(rows):
        return np.empty((0, 3), dtype=float)
    return (pts[rows[:, 0]] + pts[rows[:, 1]]) / 2.0
