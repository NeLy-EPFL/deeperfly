"""Render :class:`~deeperfly.io.PoseResult` reconstructions to MP4.

Separated from the frame-I/O layer so that pure video read/write does not pull
in matplotlib. These functions import matplotlib (and :mod:`deeperfly.viz`)
lazily, and require the ``viz`` extra.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jaxtyping import Float

from ..io import PoseResult
from .io import write_mp4


def figure_to_array(fig) -> Float[np.ndarray, "H W 3"]:
    """Rasterise a matplotlib figure to an ``(H, W, 3)`` uint8 array."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()


def render_pose3d_video(
    result: PoseResult,
    path: str | Path,
    *,
    fps: float = 30.0,
    use_smoothed: bool = True,
    elev: float = 20,
    azim: float = -60,
    background: str = "white",
    backend: str = "auto",
) -> None:
    """Render the triangulated 3D skeleton over time to an MP4.

    ``background`` is ``"white"`` or ``"black"``; limbs follow the skeleton's
    ``palette`` (legs left blue / right red).
    """
    import matplotlib.pyplot as plt

    from ..viz import matplotlib as viz

    pts3d = (
        result.pts3d_smoothed
        if (use_smoothed and result.pts3d_smoothed is not None)
        else result.pts3d
    )
    if pts3d is None:
        raise ValueError("result has no 3D points to render")
    finite = pts3d[np.isfinite(pts3d).all(-1)]
    # Fix the axis limits once from the whole sequence so the view does not jump
    # frame to frame. Use the 1st/99th percentile per axis (not min/max) so a few
    # outlier points -- a momentarily misplaced keypoint -- don't blow out the box.
    lo, hi = np.percentile(finite, [1, 99], axis=0)

    frames = []
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection="3d")
    for t in range(pts3d.shape[0]):
        ax.clear()
        viz.plot_skeleton_3d(
            pts3d[t],
            result.skeleton,
            ax=ax,
            elev=elev,
            azim=azim,
            background=background,
        )
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        # plot_skeleton_3d set the aspect from *this* frame's data, which bakes a
        # per-frame box aspect; recompute "equal" from the fixed limits so the
        # cube's shape (not just its numeric limits) stays constant across frames.
        ax.set_aspect("equal")
        frames.append(figure_to_array(fig))
    plt.close(fig)
    write_mp4(np.stack(frames), path, fps=fps, backend=backend)


def render_overlay_video(
    result: PoseResult,
    images: Float[np.ndarray, "T H W 3"],
    path: str | Path,
    *,
    camera: int = 0,
    fps: float = 30.0,
    background: str = "white",
    backend: str = "auto",
) -> None:
    """Render one camera's 2D pose overlay across frames to an MP4.

    ``background`` (``"white"`` / ``"black"``) colors the margins around the
    frame; limbs follow the skeleton's ``palette``. ``images`` may be a
    GPU-decoded tensor (e.g. when the detector ran on the GPU); it is brought to
    host NumPy here for matplotlib.
    """
    import matplotlib.pyplot as plt

    from ..viz import matplotlib as viz
    from .base import to_numpy

    images = to_numpy(images)
    frames = []
    fig, ax = plt.subplots(figsize=(images.shape[2] / 100, images.shape[1] / 100))
    for t in range(result.pts2d.shape[1]):
        ax.clear()
        conf = None if result.conf is None else result.conf[camera, t]
        viz.plot_skeleton_2d(
            result.pts2d[camera, t],
            result.skeleton,
            image=images[t],
            conf=conf,
            ax=ax,
            background=background,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        frames.append(figure_to_array(fig))
    plt.close(fig)
    write_mp4(np.stack(frames), path, fps=fps, backend=backend)
