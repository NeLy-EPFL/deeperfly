"""Frame I/O and MP4 rendering (requires the ``viz`` extra).

Reads input frames from a video file or a directory/glob of images, writes MP4s
via imageio-ffmpeg, and renders a 3D-skeleton movie (and 2D overlay movies) from
a :class:`~deeperfly.io.PoseResult`.
"""

from __future__ import annotations

import glob
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from jaxtyping import Float

from .io import PoseResult
from . import viz

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def read_video(path: str | Path) -> Float[np.ndarray, "T H W 3"]:
    """Read all frames of a video file into a ``(T, H, W, 3)`` uint8 array."""
    with imageio.get_reader(str(path)) as reader:
        return np.stack([np.asarray(frame)[..., :3] for frame in reader])


def read_images(pattern: str | Path) -> Float[np.ndarray, "T H W 3"]:
    """Read a directory (or glob) of images, sorted by name, into ``(T, H, W, 3)``."""
    p = Path(pattern)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in _IMAGE_EXTS)
    else:
        files = sorted(Path(f) for f in glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"no images matched {pattern!r}")
    return np.stack([np.asarray(imageio.imread(f))[..., :3] for f in files])


def write_mp4(
    frames: Float[np.ndarray, "T H W 3"],
    path: str | Path,
    fps: float = 30.0,
) -> None:
    """Write ``(T, H, W, 3)`` uint8 frames to an MP4 (H.264)."""
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    with imageio.get_writer(
        str(path), fps=fps, codec="libx264", macro_block_size=None
    ) as w:
        for frame in frames:
            w.append_data(frame)


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
    elev: float = 20.0,
    azim: float = -60.0,
) -> None:
    """Render the triangulated 3D skeleton over time to an MP4."""
    import matplotlib.pyplot as plt

    pts3d = (
        result.pts3d_smoothed
        if (use_smoothed and result.pts3d_smoothed is not None)
        else result.pts3d
    )
    if pts3d is None:
        raise ValueError("result has no 3D points to render")
    finite = pts3d[np.isfinite(pts3d).all(-1)]
    lo, hi = finite.min(0), finite.max(0)

    frames = []
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection="3d")
    for t in range(pts3d.shape[0]):
        ax.clear()
        viz.plot_skeleton_3d(pts3d[t], result.skeleton, ax=ax, elev=elev, azim=azim)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        frames.append(figure_to_array(fig))
    plt.close(fig)
    write_mp4(np.stack(frames), path, fps=fps)


def render_overlay_video(
    result: PoseResult,
    images: Float[np.ndarray, "T H W 3"],
    path: str | Path,
    *,
    camera: int = 0,
    fps: float = 30.0,
) -> None:
    """Render one camera's 2D pose overlay across frames to an MP4."""
    import matplotlib.pyplot as plt

    frames = []
    fig, ax = plt.subplots(figsize=(images.shape[2] / 100, images.shape[1] / 100))
    for t in range(result.pts2d.shape[1]):
        ax.clear()
        conf = None if result.conf is None else result.conf[camera, t]
        viz.plot_skeleton_2d(
            result.pts2d[camera, t], result.skeleton, image=images[t], conf=conf, ax=ax
        )
        ax.set_xticks([])
        ax.set_yticks([])
        frames.append(figure_to_array(fig))
    plt.close(fig)
    write_mp4(np.stack(frames), path, fps=fps)
