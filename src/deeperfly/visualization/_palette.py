"""Per-point RGB colors from a skeleton's limb palette, with no plotting deps.

Each point takes its limb's color from the skeleton ``palette``
(``limb_name -> hex``), falling back to ``tab10`` for limbs without an entry. The
OpenCV overlay and compositor (:mod:`deeperfly.visualization.opencv`,
:mod:`deeperfly.visualization.compose`) draw straight into image arrays, so the colors live
here as plain NumPy with no matplotlib dependency.
"""

from __future__ import annotations

import numpy as np

from ..skeleton import Skeleton

#: matplotlib's ``tab10`` as RGB floats in ``[0, 1]`` -- the fallback for limbs
#: absent from the palette (kept in sync with ``plt.get_cmap("tab10")``).
TAB10 = np.array(
    [
        (0.12156862, 0.46666666, 0.70588235),
        (1.00000000, 0.49803921, 0.05490196),
        (0.17254901, 0.62745098, 0.17254901),
        (0.83921568, 0.15294117, 0.15686274),
        (0.58039215, 0.40392156, 0.74117647),
        (0.54901960, 0.33725490, 0.29411764),
        (0.89019607, 0.46666666, 0.76078431),
        (0.49803921, 0.49803921, 0.49803921),
        (0.73725490, 0.74117647, 0.13333333),
        (0.09019607, 0.74509803, 0.81176470),
    ]
)


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    """Parse a ``#rgb`` / ``#rrggbb`` hex color to RGB floats in ``[0, 1]``.

    Parameters
    ----------
    value
        A ``#rgb`` or ``#rrggbb`` hex color string.

    Returns
    -------
    tuple of float
        The ``(r, g, b)`` channels in ``[0, 1]``.

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
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def point_colors_rgb(
    skeleton: Skeleton, palette: dict[str, str] | None = None
) -> np.ndarray:
    """``(P, 3)`` RGB floats in ``[0, 1]``, one per tracked point, by limb.

    Parameters
    ----------
    skeleton
        Skeleton supplying the per-point limb ids and the palette.
    palette
        Optional ``limb_name -> hex`` override of the skeleton palette.

    Returns
    -------
    np.ndarray
        ``(P, 3)`` RGB floats in ``[0, 1]`` (``tab10`` fallback for limbs absent
        from the palette).
    """
    palette = skeleton.palette if palette is None else palette
    out = np.empty((skeleton.n_points, 3))
    for n in range(skeleton.n_points):
        lid = int(skeleton.limb_id[n])
        name = skeleton.limb_names[lid] if 0 <= lid < len(skeleton.limb_names) else ""
        out[n] = _hex_to_rgb(palette[name]) if name in palette else TAB10[lid % 10]
    return out
