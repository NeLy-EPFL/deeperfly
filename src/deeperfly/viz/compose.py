"""Config-driven panel compositor: layer draw-ops into video frames.

A ``[[viz.videos]]`` entry describes one output video as an ordered list of
*panels* (layers) drawn onto a shared RGB buffer::

    [[viz.videos]]
    video_name = "pose3d"
    panels = [
        { plot = "imshow",      view = "rf", x0 = 0,   y0 = 0 },
        { plot = "skeleton_3d", view = "rf", x0 = 0,   y0 = 0 },
        { plot = "imshow",      view = "lf", x0 = 480, y0 = 0 },
        { plot = "skeleton_3d", view = "lf", x0 = 480, y0 = 0 },
    ]

Panels are applied in order, so a ``skeleton_*`` layer placed after an
``imshow`` of the same ``view`` / offset overlays the skeleton on that frame;
placed alone it lands on the background. This makes image-backed and
skeleton-only panels fall out of the same mechanism.

Supported ``plot`` ops:

- ``imshow``      -- the view's video frame.
- ``skeleton_2d`` -- the view's 2D detections.
- ``skeleton_3d`` -- the triangulated 3D skeleton reprojected into the view
  (OpenCV, depth-ordered).

Keyword arguments for a draw op can be set at three levels, each a table keyed
by ``plot`` op name (``skeleton_2d`` / ``skeleton_3d`` / ``imshow``) so the same
config can tune several op kinds at once::

    [viz.kwargs]                 # 1. global: every panel of every video
    skeleton_2d = { line_thickness = 2 }
    skeleton_3d = { line_thickness = 2 }

    [[viz.videos]]
    video_name = "pose3d"
    kwargs = { skeleton_3d = { point_radius = 5 } }   # 2. one video, all panels
    panels = [
        { plot = "skeleton_3d", view = "rf", line_thickness = 4 },  # 3. one panel
    ]

For each panel the three are merged in that order -- global, then this video's,
then the panel's own extra keys -- with the more specific level winning. A
level's entry for the panel's op (or the panel's extra keys) is forwarded to the
draw op (e.g. ``point_radius``, ``line_thickness``, ``palette``).

A panel's footprint comes from ``scale`` (a factor on the view's image size) or
from a target ``width`` / ``height`` in pixels -- the latter let a layout be
fixed without knowing the source image size, and take priority over ``scale``
(both -> that exact box, one -> aspect preserved, neither -> ``scale``). All
three are layout keys, settable at any of the three levels just like draw kwargs
(the panel's direct key still wins) but resized into the layer rather than
forwarded to the op. The canvas is sized to the video's own ``width`` /
``height`` when given, else to the bounding box of every panel's footprint.

The canvas background is ``black`` unless ``viz.background`` is set (global); a
single panel can override it over its own footprint with a ``background`` key::

    [viz]
    background = "white"          # canvas fill for every video
    [[viz.videos]]
    panels = [
        { plot = "skeleton_3d", view = "rf", background = "black" },  # one tile
    ]

The primitives live in :mod:`deeperfly.viz.opencv`; this module needs the
``opencv`` extra (and the ``imageio``/PyAV video stack to write MP4s).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np
from jaxtyping import Float

from . import opencv as _cv

if TYPE_CHECKING:
    from ..cameras import CameraGroup
    from ..skeleton import Skeleton

#: Panel keys consumed by the compositor itself; everything else is forwarded to
#: the draw op as keyword arguments.
_RESERVED = frozenset(
    {"plot", "view", "x0", "y0", "scale", "width", "height", "background"}
)


@dataclass
class Panel:
    """One draw-op layer: ``plot`` op for ``view`` placed at ``(x0, y0)``.

    The layer's footprint is set either by ``scale`` (resizes the view's image
    size uniformly) or by a target ``width`` / ``height`` in pixels, which lets a
    layout be fixed without knowing the source image size. ``width`` / ``height``
    take priority over ``scale``: giving both fits that exact box (axes scaled
    independently), giving one preserves aspect ratio, giving neither uses
    ``scale``. ``background``, when set, fills the panel's footprint with that
    color before the op draws (otherwise the video's canvas background shows
    through).
    """

    plot: str
    view: str
    x0: int = 0
    y0: int = 0
    scale: float = 1.0
    width: int | None = None
    height: int | None = None
    background: str | tuple[int, int, int] | None = None
    options: dict = field(default_factory=dict)

    def scales(self, view_h: int, view_w: int) -> tuple[float, float]:
        """Resolve ``(scale_x, scale_y)`` against the view's ``(height, width)``.

        ``width`` / ``height`` (target footprint pixels) take priority over
        ``scale`` -- see the class docstring.
        """
        if self.width is not None and self.height is not None:
            return self.width / view_w, self.height / view_h
        if self.width is not None:
            s = self.width / view_w
            return s, s
        if self.height is not None:
            s = self.height / view_h
            return s, s
        return self.scale, self.scale

    def footprint(self, view_h: int, view_w: int) -> tuple[int, int]:
        """The layer's ``(height, width)`` in canvas pixels at its resolved scale."""
        sx, sy = self.scales(view_h, view_w)
        return round(view_h * sy), round(view_w * sx)


@dataclass
class VideoSpec:
    """One output video: a name and an ordered list of :class:`Panel` layers.

    ``background`` is the canvas fill color (default ``"black"``); panels may
    override it over their own footprint via :attr:`Panel.background`.
    """

    video_name: str
    panels: list[Panel]
    width: int | None = None
    height: int | None = None
    background: str | tuple[int, int, int] = "black"


@dataclass
class Sources:
    """The data the panels draw from, shared across every video and frame.

    ``frames`` maps a view name to that camera's footage ``(T, H, W[, 3])``.
    ``pts2d`` / ``conf`` are aligned to ``camera_group`` order (``(V, T, N, 2)``
    / ``(V, T, N)``); ``pts3d`` is ``(T, N, 3)`` in world coordinates. Only the
    sources a video's ops actually reference need to be provided.
    """

    skeleton: "Skeleton"
    camera_group: "CameraGroup"
    frames: dict[str, np.ndarray]
    pts2d: Float[np.ndarray, "V T N 2"] | None = None
    pts3d: Float[np.ndarray, "T N 3"] | None = None
    conf: Float[np.ndarray, "V T N"] | None = None

    def _view_index(self, view: str) -> int:
        return self.camera_group.names.index(view)

    def view_size(self, view: str) -> tuple[int, int]:
        """``(height, width)`` of a view's panel, from its frames or intrinsics."""
        frames = self.frames.get(view)
        if frames is not None:
            return int(frames.shape[1]), int(frames.shape[2])
        intr = self.camera_group[view].intr  # [fx, fy, cx, cy]
        return int(round(2 * intr[3] + 1)), int(round(2 * intr[2] + 1))

    def n_frames(self) -> int:
        if self.pts3d is not None:
            return int(self.pts3d.shape[0])
        if self.pts2d is not None:
            return int(self.pts2d.shape[1])
        if self.frames:
            return int(next(iter(self.frames.values())).shape[0])
        raise ValueError("Sources has no frames, pts2d or pts3d to count")


# -- draw ops -----------------------------------------------------------------


def _op_imshow(canvas: np.ndarray, panel: Panel, src: Sources, t: int) -> None:
    scale = panel.scales(*src.view_size(panel.view))
    _cv.draw_image(canvas, src.frames[panel.view][t], panel.x0, panel.y0, scale)


def _op_skeleton_2d(canvas: np.ndarray, panel: Panel, src: Sources, t: int) -> None:
    if src.pts2d is None:
        raise ValueError("skeleton_2d panel needs Sources.pts2d")
    v = src._view_index(panel.view)
    conf = None if src.conf is None else src.conf[v, t]
    _cv.draw_skeleton_2d(
        canvas,
        src.pts2d[v, t],
        src.skeleton,
        x0=panel.x0,
        y0=panel.y0,
        scale=panel.scales(*src.view_size(panel.view)),
        conf=conf,
        **panel.options,
    )


def _op_skeleton_3d(canvas: np.ndarray, panel: Panel, src: Sources, t: int) -> None:
    if src.pts3d is None:
        raise ValueError("skeleton_3d panel needs Sources.pts3d")
    _cv.draw_skeleton_3d(
        canvas,
        src.pts3d[t],
        src.camera_group[panel.view],
        src.skeleton,
        x0=panel.x0,
        y0=panel.y0,
        scale=panel.scales(*src.view_size(panel.view)),
        **panel.options,
    )


#: ``plot`` name -> draw op. Extend to add new panel kinds.
OPS: dict[str, Callable[[np.ndarray, Panel, Sources, int], None]] = {
    "imshow": _op_imshow,
    "skeleton_2d": _op_skeleton_2d,
    "skeleton_3d": _op_skeleton_3d,
}


# -- config parsing -----------------------------------------------------------


def _op_kwargs(table: dict, plot: str) -> dict:
    """Look up a per-op kwargs table's entry for ``plot`` (a dict, else empty)."""
    value = table.get(plot, {})
    if not isinstance(value, dict):
        raise ValueError(
            f"viz kwargs for plot op {plot!r} must be a table of keyword "
            f"arguments, got {value!r}"
        )
    return value


def _layout_key(panel: dict, options: dict, key: str):
    """Resolve a structural layout key (``scale`` / ``width`` / ``height``).

    A direct key on the panel wins over one merged in from the op-kwargs levels.
    The key is popped from ``options`` either way so it is never forwarded to the
    draw op (it resizes the layer, it is not a draw argument).
    """
    value = panel[key] if key in panel else options.get(key)
    options.pop(key, None)
    return value


def read_video_specs(config: dict | str | Path) -> list[VideoSpec]:
    """Parse ``[[viz.videos]]`` from a config dict or TOML path into specs.

    Per-op keyword arguments are resolved here and baked into each panel's
    ``options``, merged from least to most specific: ``[viz.kwargs]`` (global),
    the video entry's ``kwargs`` (all its panels), then the panel's own extra
    keys. Each ``kwargs`` table is keyed by ``plot`` op name. The layout keys
    ``scale`` / ``width`` / ``height`` taken from those kwargs are lifted onto the
    matching :class:`Panel` fields rather than forwarded. The canvas background
    comes from ``viz.background`` (default ``"black"``); a panel's ``background``
    key sets :attr:`Panel.background`.
    """
    if not isinstance(config, dict):
        with open(config, "rb") as f:
            config = tomllib.load(f)
    viz = config.get("viz", {})
    global_kwargs = viz.get("kwargs", {})
    background = viz.get("background", "black")
    specs: list[VideoSpec] = []
    for entry in viz.get("videos", []):
        video_kwargs = entry.get("kwargs", {})
        panels = []
        for p in entry.get("panels", []):
            plot = p["plot"]
            options = {
                **_op_kwargs(global_kwargs, plot),
                **_op_kwargs(video_kwargs, plot),
                **{k: v for k, v in p.items() if k not in _RESERVED},
            }
            # scale / width / height resize the layer rather than reaching the
            # draw op, so pull them out of the merged kwargs (a panel may set them
            # per-op via the kwargs levels or directly, the direct key winning).
            scale = _layout_key(p, options, "scale")
            width = _layout_key(p, options, "width")
            height = _layout_key(p, options, "height")
            panels.append(
                Panel(
                    plot=plot,
                    view=p["view"],
                    x0=int(p.get("x0", 0)),
                    y0=int(p.get("y0", 0)),
                    scale=1.0 if scale is None else float(scale),
                    width=None if width is None else int(width),
                    height=None if height is None else int(height),
                    background=p.get("background"),
                    options=options,
                )
            )
        specs.append(
            VideoSpec(
                video_name=entry["video_name"],
                panels=panels,
                width=entry.get("width"),
                height=entry.get("height"),
                background=background,
            )
        )
    return specs


# -- rendering ----------------------------------------------------------------


def canvas_size(spec: VideoSpec, src: Sources) -> tuple[int, int]:
    """``(height, width)`` for ``spec``: explicit when set, else panel bbox."""
    height, width = spec.height, spec.width
    if height is None or width is None:
        bbox_h = bbox_w = 0
        for panel in spec.panels:
            ph, pw = panel.footprint(*src.view_size(panel.view))
            bbox_h = max(bbox_h, panel.y0 + ph)
            bbox_w = max(bbox_w, panel.x0 + pw)
        height = bbox_h if height is None else height
        width = bbox_w if width is None else width
    return int(height), int(width)


def compose_frame(spec: VideoSpec, src: Sources, t: int) -> np.ndarray:
    """Composite frame ``t`` of ``spec`` into a single RGB array."""
    height, width = canvas_size(spec, src)
    canvas = _cv.new_canvas(height, width, spec.background)
    for panel in spec.panels:
        try:
            op = OPS[panel.plot]
        except KeyError:
            raise ValueError(
                f"unknown plot op {panel.plot!r}; choose from {sorted(OPS)}"
            ) from None
        if panel.background is not None:
            ph, pw = panel.footprint(*src.view_size(panel.view))
            _cv.fill_region(canvas, panel.x0, panel.y0, pw, ph, panel.background)
        op(canvas, panel, src, t)
    return canvas


def render_video(
    spec: VideoSpec,
    src: Sources,
    *,
    n_frames: int | None = None,
    progress: Callable[[Iterable[int]], Iterable[int]] | None = None,
) -> Float[np.ndarray, "T H W 3"]:
    """Composite every frame of ``spec`` into a ``(T, H, W, 3)`` uint8 stack.

    ``progress`` optionally wraps the per-frame iterator (e.g. a rich progress
    bar) so callers can show progress while compositing; it defaults to the
    identity, keeping this library UI-free.
    """
    n = src.n_frames() if n_frames is None else n_frames
    steps = range(n) if progress is None else progress(range(n))
    return np.stack([compose_frame(spec, src, t) for t in steps])


def render_videos(
    config: dict | str | Path,
    src: Sources,
    outdir: str | Path,
    *,
    fps: float = 30.0,
    backend: str = "auto",
) -> list[Path]:
    """Render every ``[[viz.videos]]`` in ``config`` to ``<outdir>/<name>.mp4``."""
    from ..video import write_mp4

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for spec in read_video_specs(config):
        frames = render_video(spec, src)
        path = outdir / f"{spec.video_name}.mp4"
        write_mp4(frames, path, fps=fps, backend=backend)
        paths.append(path)
    return paths
