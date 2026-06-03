"""Headless matplotlib visualization of 2D overlays and 3D skeletons.

Functions draw onto an existing ``Axes`` when given one (so callers compose
montages / video frames) or create a figure otherwise. Bones and per-limb
colors come from the :class:`~deeperfly.skeleton.Skeleton` (its ``palette``);
detector confidence (when supplied) modulates joint opacity. Requires the
``viz`` extra (``matplotlib``); import this module only when plotting.

The faster, figure-less :mod:`deeperfly.viz.opencv` backend draws the same
overlays straight into image arrays -- prefer it for video.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=False)  # default to a headless backend
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np  # noqa: E402
from jaxtyping import Float  # noqa: E402

from ..skeleton import Skeleton  # noqa: E402

#: Background presets: figure/axes face color and the matching foreground
#: (spines, ticks, labels, 3D panes) so plots read on white or black.
BACKGROUNDS: dict[str, dict[str, str]] = {
    "white": {"face": "white", "fg": "black"},
    "black": {"face": "black", "fg": "white"},
}


def limb_colors(
    skeleton: Skeleton, *, palette: dict[str, str] | None = None
) -> np.ndarray:
    """A stable RGBA color per tracked point, colored by its limb.

    Colors come from the skeleton's :attr:`~deeperfly.skeleton.Skeleton.palette`
    (a ``limb_name -> hex`` mapping; pass ``palette`` to override it). Limbs
    without an entry fall back to ``tab10`` so non-fly skeletons stay
    distinguishable.
    """
    palette = skeleton.palette if palette is None else palette
    cmap = plt.get_cmap("tab10")
    colors = []
    for n in range(skeleton.n_points):
        lid = int(skeleton.limb_id[n])
        name = skeleton.limb_names[lid] if lid < len(skeleton.limb_names) else ""
        if name in palette:
            colors.append(mcolors.to_rgba(palette[name]))
        else:
            colors.append(cmap((lid % 10) / 10.0))
    return np.array(colors)


def apply_background(ax: Axes | Axes3D, background: str = "white") -> Axes | Axes3D:
    """Style ``ax`` (and its figure) for a ``"white"`` or ``"black"`` background.

    Sets the figure and axes face colors and recolors spines / ticks / labels
    (and the panes + grid for 3D axes) to a contrasting foreground. Re-apply after
    ``ax.clear()`` -- the plotting helpers below do this for you.
    """
    try:
        theme = BACKGROUNDS[background]
    except KeyError:
        raise ValueError(
            f"background must be one of {sorted(BACKGROUNDS)}; got {background!r}"
        ) from None
    face, fg = theme["face"], theme["fg"]
    ax.get_figure().set_facecolor(face)
    ax.set_facecolor(face)
    if hasattr(ax, "zaxis"):  # 3D
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(mcolors.to_rgba(face))
            axis.line.set_color(fg)
            axis.label.set_color(fg)
        ax.tick_params(colors=fg)
        ax.grid(color=fg, alpha=0.15)
    else:
        for spine in ax.spines.values():
            spine.set_color(fg)
        ax.tick_params(colors=fg)
        ax.title.set_color(fg)
    return ax


def plot_skeleton_2d(
    pts2d: Float[np.ndarray, "N 2"],
    skeleton: Skeleton,
    *,
    image: Float[np.ndarray, "H W 3"] | None = None,
    conf: Float[np.ndarray, "N"] | None = None,
    ax: Axes | None = None,
    point_size: float = 12.0,
    background: str = "white",
) -> Axes:
    """Draw one camera's 2D joints + bones, optionally over its image.

    ``pts2d`` is a single frame/camera ``(N, 2)`` in pixels; NaN joints (and the
    bones touching them) are skipped. ``background`` is ``"white"`` or ``"black"``
    (matters around the image margins, and fully when ``image`` is ``None``).
    """
    if ax is None:
        _, ax = plt.subplots()
    apply_background(ax, background)
    if image is not None:
        ax.imshow(np.asarray(image), cmap="gray")
    pts2d = np.asarray(pts2d, dtype=float)
    colors = limb_colors(skeleton)

    for a, b in skeleton.bones:
        if np.isfinite(pts2d[[a, b]]).all():
            ax.plot(
                pts2d[[a, b], 0], pts2d[[a, b], 1], "-", color=colors[a], linewidth=1.0
            )
    finite = np.isfinite(pts2d).all(-1)
    alpha = np.ones(skeleton.n_points) if conf is None else np.clip(conf, 0, 1)
    for n in np.flatnonzero(finite):
        ax.scatter(
            pts2d[n, 0],
            pts2d[n, 1],
            s=point_size,
            color=colors[n],
            alpha=float(alpha[n]),
        )
    ax.set_aspect("equal")
    return ax


def plot_skeleton_3d(
    pts3d: Float[np.ndarray, "N 3"],
    skeleton: Skeleton,
    *,
    ax: Axes3D | None = None,
    elev: float = 20.0,
    azim: float = -60.0,
    background: str = "white",
) -> Axes3D:
    """Draw a 3D skeleton (bones) into a 3D axis.

    ``background`` is ``"white"`` or ``"black"`` (sets the panes, grid and labels).
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    apply_background(ax, background)
    pts3d = np.asarray(pts3d, dtype=float)
    colors = limb_colors(skeleton)

    for a, b in skeleton.bones:
        if np.isfinite(pts3d[[a, b]]).all():
            ax.plot(*pts3d[[a, b]].T, "-", color=colors[a], linewidth=1.0)
    finite = np.isfinite(pts3d).all(-1)
    ax.scatter(*pts3d[finite].T, s=10, c=colors[finite])
    ax.set_aspect("equal")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.axis("off")
    return ax


def overlay_grid(
    pts2d: Float[np.ndarray, "V N 2"],
    skeleton: Skeleton,
    *,
    images: list[Float[np.ndarray, "H W 3"]] | None = None,
    camera_names: list[str] | None = None,
    ncols: int = 4,
    background: str = "white",
) -> Figure:
    """A montage of every camera's 2D overlay for a single frame."""
    n_views = pts2d.shape[0]
    nrows = int(np.ceil(n_views / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3 * ncols, 2 * nrows), squeeze=False
    )
    fig.set_facecolor(BACKGROUNDS.get(background, BACKGROUNDS["white"])["face"])
    for v in range(nrows * ncols):
        ax = axes[v // ncols][v % ncols]
        if v >= n_views:
            apply_background(ax, background)
            ax.axis("off")
            continue
        img = None if images is None else images[v]
        plot_skeleton_2d(pts2d[v], skeleton, image=img, ax=ax, background=background)
        if camera_names is not None:
            ax.set_title(camera_names[v], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig
