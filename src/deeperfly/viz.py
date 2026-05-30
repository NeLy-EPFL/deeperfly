"""Headless matplotlib visualisation of 2D overlays and 3D skeletons.

Functions draw onto an existing ``Axes`` when given one (so callers compose
montages / video frames) or create a figure otherwise. Bones come from the
:class:`~deeperfly.skeleton.Skeleton`; each limb gets a stable colour and
detector confidence (when supplied) modulates joint opacity. Requires the
``viz`` extra (``matplotlib``); import this module only when plotting.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=False)  # default to a headless backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from jaxtyping import Float  # noqa: E402

from .skeleton import Skeleton  # noqa: E402


def limb_colors(skeleton: Skeleton) -> np.ndarray:
    """A stable RGBA colour per tracked point, coloured by its limb."""
    cmap = plt.get_cmap("tab10")
    return np.array([cmap((lid % 10) / 10.0) for lid in skeleton.limb_id])


def plot_skeleton_2d(
    pts2d: Float[np.ndarray, "N 2"],
    skeleton: Skeleton,
    *,
    image: Float[np.ndarray, "H W 3"] | None = None,
    conf: Float[np.ndarray, "N"] | None = None,
    ax: plt.Axes | None = None,
    point_size: float = 12.0,
) -> plt.Axes:
    """Draw one camera's 2D joints + bones, optionally over its image.

    ``pts2d`` is a single frame/camera ``(N, 2)`` in pixels; NaN joints (and the
    bones touching them) are skipped.
    """
    if ax is None:
        _, ax = plt.subplots()
    if image is not None:
        ax.imshow(np.asarray(image))
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
    ax: plt.Axes | None = None,
    elev: float = 20.0,
    azim: float = -60.0,
    draw_bones3d: bool = True,
) -> plt.Axes:
    """Draw a 3D skeleton (bones + cross-body bones) into a 3D axis."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    pts3d = np.asarray(pts3d, dtype=float)
    colors = limb_colors(skeleton)

    edges = skeleton.bones
    if draw_bones3d and skeleton.bones3d.size:
        edges = np.concatenate([skeleton.bones, skeleton.bones3d], axis=0)
    for a, b in edges:
        if np.isfinite(pts3d[[a, b]]).all():
            ax.plot(*pts3d[[a, b]].T, "-", color=colors[a], linewidth=1.0)
    finite = np.isfinite(pts3d).all(-1)
    ax.scatter(*pts3d[finite].T, s=10, c=colors[finite])
    ax.view_init(elev=elev, azim=azim)
    return ax


def overlay_grid(
    pts2d: Float[np.ndarray, "V N 2"],
    skeleton: Skeleton,
    *,
    images: list[Float[np.ndarray, "H W 3"]] | None = None,
    camera_names: list[str] | None = None,
    ncols: int = 4,
) -> plt.Figure:
    """A montage of every camera's 2D overlay for a single frame."""
    n_views = pts2d.shape[0]
    nrows = int(np.ceil(n_views / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3 * ncols, 2 * nrows), squeeze=False
    )
    for v in range(nrows * ncols):
        ax = axes[v // ncols][v % ncols]
        if v >= n_views:
            ax.axis("off")
            continue
        img = None if images is None else images[v]
        plot_skeleton_2d(pts2d[v], skeleton, image=img, ax=ax)
        if camera_names is not None:
            ax.set_title(camera_names[v], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig
