"""Config-driven panel compositor: layer draw-ops into video frames.

A ``[[visualization.videos]]`` entry describes one output video as an
ordered list of *panels* (layers) drawn onto a shared RGB buffer::

    [[visualization.videos]]
    video_name = "pose3d"
    panels = [
        { plot = "imshow",      view = "rf", x0 = 0,   y0 = 0 },
        { plot = "skeleton_3d", view = "rf", x0 = 0,   y0 = 0 },
        { plot = "imshow",      view = "lf", x0 = 480, y0 = 0 },
        { plot = "skeleton_3d", view = "lf", x0 = 480, y0 = 0 },
    ]

Panels are applied in order, so a ``skeleton_*`` layer after an ``imshow`` of the
same ``view`` / offset overlays the skeleton; placed alone it lands on the
background. Supported ``plot`` ops:

- ``imshow``      -- the view's video frame.
- ``skeleton_2d`` -- the view's 2D detections.
- ``skeleton_3d`` -- the 3D skeleton reprojected into the view (OpenCV, depth-ordered).

Two panel keys shape what a cell contains rather than what is drawn in it:

- ``crop = [x, y, w, h]`` shows that window of the view's raw frame instead of the whole
  thing -- for an axial camera whose animal is a small part of a wide frame, the
  difference between a judgeable panel and a smudge. The picture and the geometry move
  together (2D points by ``-(x, y)``, the camera's principal point too).
- ``crop = "pose2d"`` takes that window from the detector instead of restating it: the
  panel shows what the ``[[pose2d.pathways]]`` feeding this view detected through. Set it
  once under ``[visualization]`` and every panel follows its own view, so the crop lives
  in exactly one place in the config -- which matters because a crop is per RECORDING
  *regenerates* the ``[pose2d]`` crops per recording, and a hand-copied panel box would
  silently keep showing the old window. ``crop = "<preprocessor name>"`` borrows one
  named ``[[pose2d.preprocessors]]`` chain explicitly. Settable at all three levels
  (global / video / panel), most specific winning.
- ``clip`` (default true) keeps a panel inside its footprint. The draw ops honour
  ``(x0, y0)`` but do not stop at the panel edge, so without it a limb projecting out of
  its cell is painted over the neighbouring camera.

A panel may also name the reserved view ``"bird"``: a dorsal plan view derived from the
3D pose (:mod:`deeperfly.visualization.bird`). It has no footage -- give it a
``skeleton_3d`` panel and no ``imshow``. A rig camera of that name takes precedence.

Draw-op kwargs merge across three levels, each a table keyed by ``plot`` op name,
most specific winning::

    [visualization.kwargs]   # 1. global: every panel of every video
    skeleton_3d = { line_thickness = 2 }

    [[visualization.videos]]
    video_name = "pose3d"
    kwargs = { skeleton_3d = { point_radius = 5 } }   # 2. one video
    panels = [
        { plot = "skeleton_3d", view = "rf", line_thickness = 4 },  # 3. one panel
    ]

The layout keys ``scale`` / ``width`` / ``height`` are settable at the same levels
but resize the layer instead of reaching the op (``width`` / ``height`` win over
``scale``: both -> that exact box, one -> aspect preserved). The canvas is sized
to the video's ``width`` / ``height`` when given, else the panels' bounding box.
Its background is ``black`` unless ``visualization.background`` is set; a
panel's ``background`` key repaints just its tile.

Primitives live in :mod:`deeperfly.visualization.opencv`; MP4 writing uses the PyAV stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Iterator, Protocol

import numpy as np
from jaxtyping import Float

from . import opencv as _cv
from .bird import BIRD_VIEW as _BIRD_VIEW

if TYPE_CHECKING:
    from ..cameras import Camera, CameraGroup
    from ..config import Config
    from ..preprocessing import FrameTransform
    from ..skeleton import Skeleton

log = logging.getLogger("deeperfly")

#: The value of a panel's ``crop`` that means "the window the detector looked through",
#: resolved per view from ``[[pose2d.pathways]]``. Reserved: a ``[[pose2d.preprocessors]]``
#: of this name cannot be referenced by name.
DETECTOR_CROP = "pose2d"

#: Panel keys consumed by the compositor itself; everything else is forwarded to
#: the draw op as keyword arguments.
_RESERVED = frozenset(
    {
        "plot",
        "view",
        "x0",
        "y0",
        "scale",
        "width",
        "height",
        "background",
        "crop",
        "clip",
        "stage",
    }
)


@dataclass
class Panel:
    """One draw-op layer: ``plot`` op for ``view`` placed at ``(x0, y0)``.

    The footprint is set by ``scale`` (uniform) or a target ``width`` / ``height``
    in pixels, the latter winning: both -> that exact box, one -> aspect preserved.
    ``background``, when set, fills the footprint with that color before the op
    draws (otherwise the canvas background shows through).

    ``crop`` is ``(x, y, w, h)`` in the view's RAW footage pixels: the panel then shows
    that window of the camera instead of the whole frame, and the layer's footprint is
    computed against the window. It moves the image origin, so everything geometric
    moves with it -- the 2D points by ``-(x, y)`` and the camera's principal point too --
    which is the whole reason it is a panel key rather than something a caller does to
    the frames beforehand. A cropped picture drawn under an uncropped projection is a
    plausible-looking overlay that is simply in the wrong place.

    ``crop_from`` is the same window *borrowed* rather than written down: the
    preprocessing chain(s) a config's ``crop = "pose2d"`` referred to, whose window
    becomes this panel's ``crop``. The two are mutually exclusive, and the borrowed one
    is resolved late, by :meth:`resolve_crop` -- see there for why it cannot be resolved
    when the config is parsed.

    ``clip`` (default true) confines the op to the footprint. Without it the draw ops
    honour ``(x0, y0)`` but keep going past the panel, so in a grid a leg projecting out
    of its cell is painted across the neighbouring camera's picture.

    ``stage`` names the pipeline stage whose points to draw -- ``"pose2d"``,
    ``"triangulation"``, ``"eks"``, ``"pictorial_structures"``. Unset means "whatever the
    result resolved to", which is the MOST DERIVED stage present, so a video's meaning
    changes when a later stage is enabled: turn on ``[pipeline] eks`` and a panel that used to
    show the triangulation shows the smoother's output instead, under the same video
    name. Naming the stage is how a video keeps meaning one thing, and how a before/after
    pair is expressed at all.
    """

    plot: str
    view: str
    x0: int = 0
    y0: int = 0
    scale: float = 1.0
    width: int | None = None
    height: int | None = None
    background: str | tuple[int, int, int] | None = None
    crop: tuple[int, int, int, int] | None = None
    clip: bool = True
    stage: str | None = None
    crop_from: tuple["FrameTransform", ...] = ()
    options: dict = field(default_factory=dict)

    def resolve_crop(self, src: "Sources") -> "Panel":
        """This panel with any borrowed ``crop_from`` turned into a concrete ``crop``.

        Deferred to here rather than done while parsing the config because the window a
        preprocessing chain looks through depends on the RAW frame size (a flip's pixel
        map is ``-x + (w - 1)``), and a config carries no frame sizes -- they are read off
        the footage, which is what :class:`Sources` holds. Idempotent, and free for a
        panel that borrows nothing.
        """
        if not self.crop_from:
            return self
        from dataclasses import replace as _replace

        return _replace(self, crop=src.window(self.view, self.crop_from), crop_from=())

    def scales(self, view_h: int, view_w: int) -> tuple[float, float]:
        """Resolve ``(scale_x, scale_y)`` against the view's ``(height, width)``.

        ``width`` / ``height`` (target footprint pixels) take priority over
        ``scale`` -- see the class docstring.

        Parameters
        ----------
        view_h, view_w
            The source view's height and width in pixels.

        Returns
        -------
        scale_x, scale_y : float
            The per-axis resize factors.
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
        """The layer's ``(height, width)`` in canvas pixels at its resolved scale.

        Parameters
        ----------
        view_h, view_w
            The source view's height and width in pixels.

        Returns
        -------
        height, width : int
            The footprint in canvas pixels.
        """
        sx, sy = self.scales(view_h, view_w)
        return round(view_h * sy), round(view_w * sx)


@dataclass
class VideoSpec:
    """One output video: a name and an ordered list of :class:`Panel` layers.

    ``background`` is the canvas fill (default ``"black"``); panels may override it
    via :attr:`Panel.background`. The output frame rate is set by one of
    ``output_fps`` (explicit) or ``speed`` (a multiple of the recording's fps;
    ``0.5`` is slow motion); both ``None`` plays at the native rate
    (:meth:`resolve_fps`).
    """

    video_name: str
    panels: list[Panel]
    width: int | None = None
    height: int | None = None
    background: str | tuple[int, int, int] = "black"
    output_fps: float | None = None
    speed: float | None = None

    def resolve_fps(self, input_fps: float) -> float:
        """Concrete output fps from ``output_fps`` / ``speed`` and the input fps.

        An explicit ``output_fps`` wins; otherwise ``speed`` scales the input
        recording's frame rate (``input_fps * speed``); with neither, the output
        plays at the input rate.

        Parameters
        ----------
        input_fps
            The recording's frame rate.

        Returns
        -------
        float
            The concrete output frame rate.
        """
        if self.output_fps is not None:
            return float(self.output_fps)
        if self.speed is not None:
            return float(input_fps) * float(self.speed)
        return float(input_fps)


class FrameSeq(Protocol):
    """A view's footage: how many frames, how big, and the one being drawn.

    Deliberately narrow -- those three questions are *all* the compositor ever asks of
    footage, and stating that is what lets a caller hand over something other than a
    resident array. A ``(T, H, W[, 3])`` ndarray satisfies it, and so does
    :class:`~deeperfly.io.CursorFrames`, which answers ``frames[t]`` by decoding that one
    frame. The pipeline passes the latter: a render never holds more than its look-ahead,
    so materializing the clip to satisfy a type would reintroduce a cost nothing here
    needs (eight 1984x512 cameras over 5900 frames is 155 GB).

    Frames are read in near-sequential forward order (see :func:`_composited_in_order`),
    which a lazy implementation may rely on -- and ``__getitem__`` is only ever called
    with a single integer, never a slice or a fancy index.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """``(T, H, W[, 3])`` -- frame count first, then one frame's shape."""
        ...

    def __getitem__(self, t: int, /) -> np.ndarray:
        """One frame, ``(H, W, 3)`` uint8 RGB (or ``(H, W)`` if it has no color)."""
        ...


@dataclass
class Sources:
    """The data the panels draw from, shared across every video and frame.

    ``frames`` maps a view name to that camera's footage as a :class:`FrameSeq`
    (``(T, H, W[, 3])``) -- an ndarray, or anything that serves one frame at a time.
    ``pts2d`` / ``conf`` are aligned to ``camera_group`` order (``(V, T, P, 2)``
    / ``(V, T, P)``); ``pts3d`` is ``(T, P, 3)`` in world coordinates. Only the
    sources a video's ops actually reference need to be provided.
    """

    skeleton: "Skeleton"
    camera_group: "CameraGroup"
    frames: dict[str, FrameSeq]
    pts2d: Float[np.ndarray, "V T P 2"] | None = None
    pts3d: Float[np.ndarray, "T P 3"] | None = None
    conf: Float[np.ndarray, "V T P"] | None = None
    nmf_pts3d: Float[np.ndarray, "T P 3"] | None = None
    nmf_angles: Float[np.ndarray, "T D"] | None = None
    nmf_angle_names: list[str] | None = None
    nmf_head_scale: float = 1.0
    nmf_abdomen_scale: float = 1.0
    #: ``chain name -> (3,)`` model-unit shift of a chain's base onto its measured
    #: landmark, from the IK stage. The angles were fitted about the shifted pivot, so
    #: the overlay has to be drawn about it too.
    nmf_chain_offsets: dict[str, np.ndarray] = field(default_factory=dict)
    nmf_body_scale: float = 1.0
    nmf_hide_parts: tuple[str, ...] = ("wings",)
    #: Per-stage points, for panels that name a ``stage`` instead of taking whatever the
    #: result resolved to. Keyed by stage name (``"pose2d"``, ``"triangulation"``,
    #: ``"eks"``, ...); a stage absent from the file is simply absent here, and asking
    #: for it is an error rather than a silent fall back to the default -- falling back
    #: would render a before/after pair as two copies of the same thing.
    stage_pts2d: dict[str, np.ndarray] = field(default_factory=dict)
    stage_pts3d: dict[str, np.ndarray] = field(default_factory=dict)
    _pose_cache: dict = field(default_factory=dict, repr=False, compare=False)
    #: Kept apart from ``_pose_cache``, which is cleared every frame.
    _window_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def _view_index(self, view: str) -> int:
        return self.camera_group.names.index(view)

    def window(
        self, view: str, transforms: tuple["FrameTransform", ...]
    ) -> tuple[int, int, int, int] | None:
        """The raw-frame crop box a borrowed preprocessing chain looks through.

        ``None`` means the whole frame, so an uncropped view keeps the no-crop path (no
        slicing, no shifted principal point) instead of a full-frame box that would mean
        the same thing more expensively.

        Parameters
        ----------
        view
            The camera name, whose raw footage size anchors the chain.
        transforms
            The candidate chains (see
            :meth:`~deeperfly.pose2d.pathways.DetectionPlan.view_transforms`). Several
            arise when a view is fed by more than one pathway -- typically the rig's
            mirrored twin, which looks through the same window -- so they are required to
            *agree on the window*, not to be the same chain.

        Returns
        -------
        tuple or None
            The ``(x, y, width, height)`` window, or ``None`` for the whole frame.

        Raises
        ------
        ValueError
            If the candidates disagree on the window: which one the panel meant is then a
            real question about the config, and guessing would put the picture and the
            detector's box out of step -- silently, since either window renders fine.
        """
        key = (view, transforms)
        if key in self._window_cache:
            return self._window_cache[key]
        raw_size = self.view_size(view)  # the uncropped view: frames, else intrinsics
        boxes = {t.raw_window(raw_size) for t in transforms}
        if len(boxes) > 1:
            raise ValueError(
                f"view {view!r} is fed by pathways that detect through different "
                f"windows ({sorted(boxes)}); a panel cannot show both. Write the panel's "
                f"crop as an explicit [x, y, width, height], or name the one "
                f"[[pose2d.preprocessors]] you mean."
            )
        box = boxes.pop() if boxes else None
        if box == (0, 0, raw_size[1], raw_size[0]):
            box = None  # the whole frame
        self._window_cache[key] = box
        return box

    def nmf_posed(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """The posed mesh ``(verts, drawable_faces)`` for frame ``t`` (data-estimated scale).

        Memoised on the most recent frame so every view of that frame reuses one
        pose -- the GPU rasterizer in turn reuses one geometry upload across the
        cameras (it keys on the vertex array's identity).

        The configured hidden parts are subtracted **here**, not per panel, and that is what
        makes the sentence above true. The rasterizer keys its frame cache on the identity of
        *both* arrays it is given, so a panel that recomputed ``valid & ~hidden`` for itself
        handed over a new object every time and missed the cache on every view -- paying for
        the vertex normals, the index build and three buffer uploads eight times a frame for
        one frame's geometry. Same faces either way; only the number of times they are
        computed differs.
        """
        key = (
            int(t),
            self.nmf_head_scale,
            self.nmf_abdomen_scale,
            self.nmf_body_scale,
            self.nmf_hide_parts,
            tuple(
                sorted(
                    (k, tuple(float(x) for x in np.asarray(v).reshape(3)))
                    for k, v in self.nmf_chain_offsets.items()
                )
            ),
        )
        hit = self._pose_cache.get(key)
        if hit is not None:
            return hit
        from ..inverse_kinematics.mesh import load_nmf_mesh

        mesh = load_nmf_mesh()
        angles = None if self.nmf_angles is None else self.nmf_angles[t]
        verts, valid = mesh.pose(
            self.nmf_pts3d[t],
            angles,
            self.nmf_angle_names,
            head_scale=self.nmf_head_scale,
            abdomen_scale=self.nmf_abdomen_scale,
            chain_offsets=self.nmf_chain_offsets,
            body_scale=self.nmf_body_scale,
        )
        posed = (verts, valid & ~mesh.hidden_face_mask(self.nmf_hide_parts))
        self._pose_cache.clear()  # only the current frame's pose is reused
        self._pose_cache[key] = posed
        return posed

    def view_size(
        self, view: str, crop: tuple[int, int, int, int] | None = None
    ) -> tuple[int, int]:
        """``(height, width)`` of a view's panel, from its crop, frames or intrinsics.

        Parameters
        ----------
        view
            The camera name.
        crop
            The panel's ``(x, y, w, h)`` window, when it has one -- then the panel is
            the size of the window, not of the camera.

        Returns
        -------
        height, width : int
            The view size.
        """
        if crop is not None:
            return int(crop[3]), int(crop[2])
        frames = self.frames.get(view)
        if frames is not None:
            return int(frames.shape[1]), int(frames.shape[2])
        intr = self.camera(view).intr  # [fx, fy, cx, cy]
        return int(round(2 * intr[3] + 1)), int(round(2 * intr[2] + 1))

    def camera(
        self, view: str, crop: tuple[int, int, int, int] | None = None
    ) -> "Camera":
        """The camera a panel projects through -- cropped, and possibly synthetic.

        A rig camera always wins. Only when the rig has no such view does the reserved
        name :data:`~deeperfly.visualization.bird.BIRD_VIEW` resolve to a dorsal plan
        view derived from ``pts3d`` (built once and memoised, since fitting it reads
        the whole clip).

        ``crop`` shifts the principal point by ``-(x, y)``: the crop moved the image
        origin, and the projection has to move with the picture.
        """
        if view in self.camera_group.names:
            cam = self.camera_group[view]
        elif view == _BIRD_VIEW:
            cam = self._bird_camera()
        else:
            raise ValueError(
                f"panel view {view!r} is not in the rig ({self.camera_group.names}) "
                f"and is not the derived {_BIRD_VIEW!r} view"
            )
        if crop is None:
            return cam
        from dataclasses import replace as _replace

        intr = np.asarray(cam.intr, dtype=float).copy()
        intr[2] -= float(crop[0])
        intr[3] -= float(crop[1])
        return _replace(cam, intr=intr)

    def _bird_camera(self) -> "Camera":
        """The derived plan view, fitted ONCE and shared by every panel that asks.

        Fitted from the default ``pts3d`` even when a panel names a different stage, so
        a before/after pair frames identically and the only thing that moves on screen
        is the pose -- which is the entire point of rendering the pair.
        """
        cached = self._pose_cache.get("__bird__")
        if cached is None:
            from .bird import dorsal_camera

            pts3d = self.pts3d
            if pts3d is None:  # a caller that supplied only per-stage arrays
                pts3d = next(iter(self.stage_pts3d.values()), None)
            if pts3d is None:
                raise ValueError(
                    f"the {_BIRD_VIEW!r} panel is a plan view derived from the 3D pose, "
                    "so it needs Sources.pts3d (enable [pipeline] triangulation)"
                )
            like = next(iter(self.camera_group), None)
            cached = dorsal_camera(pts3d, self.skeleton, like=like)
            self._pose_cache["__bird__"] = cached
        return cached

    def frame(
        self, view: str, t: int, crop: tuple[int, int, int, int] | None = None
    ) -> np.ndarray:
        """Frame ``t`` of a view's footage, windowed to ``crop`` when it has one."""
        frames = self.frames.get(view)
        if frames is None:
            raise ValueError(
                f"no footage for view {view!r}"
                + (
                    f" -- {_BIRD_VIEW!r} is a synthetic viewpoint with no camera behind "
                    "it, so it takes a skeleton_3d panel and no imshow"
                    if view == _BIRD_VIEW
                    else ""
                )
            )
        image = frames[t]
        if crop is None:
            return image
        x, y, w, h = (int(v) for v in crop)
        return image[y : y + h, x : x + w]

    def _staged(self, stage: str | None, which: str):
        """One stage's array, or the result's own resolved one when ``stage`` is None."""
        if stage is None:
            return getattr(self, "pts2d" if which == "2d" else "pts3d")
        table = self.stage_pts2d if which == "2d" else self.stage_pts3d
        got = table.get(stage)
        if got is None:
            raise ValueError(
                f"no {which} points for stage {stage!r} (have: "
                f"{sorted(table) or 'none'}). A panel that names a stage draws THAT "
                "stage or nothing -- falling back would silently render a before/after "
                "pair as two copies of the same array."
            )
        return got

    def points3d(self, t: int, stage: str | None = None) -> np.ndarray:
        """The 3D pose for frame ``t``, from ``stage`` when the panel names one."""
        pts = self._staged(stage, "3d")
        if pts is None:
            raise ValueError("this panel needs Sources.pts3d")
        return pts[t]

    def points2d(
        self,
        view: str,
        t: int,
        crop: tuple[int, int, int, int] | None = None,
        stage: str | None = None,
    ) -> np.ndarray:
        """The view's 2D points for frame ``t``, in the panel's own pixel frame."""
        pts = self._staged(stage, "2d")
        if pts is None:
            raise ValueError("this panel needs Sources.pts2d")
        pts = pts[self._view_index(view), t]
        if crop is None:
            return pts
        return pts - np.asarray([crop[0], crop[1]], dtype=float)

    def n_frames(self) -> int:
        if self.pts3d is not None:
            return int(self.pts3d.shape[0])
        if self.nmf_pts3d is not None:
            return int(self.nmf_pts3d.shape[0])
        if self.pts2d is not None:
            return int(self.pts2d.shape[1])
        if self.frames:
            return int(next(iter(self.frames.values())).shape[0])
        raise ValueError("Sources has no frames, pts2d or pts3d to count")


# -- draw ops -----------------------------------------------------------------


def _op_imshow(canvas: np.ndarray, panel: Panel, src: Sources, t: int) -> None:
    scale = panel.scales(*src.view_size(panel.view, panel.crop))
    _cv.draw_image(
        canvas, src.frame(panel.view, t, panel.crop), panel.x0, panel.y0, scale
    )


def _op_skeleton_2d(canvas: np.ndarray, panel: Panel, src: Sources, t: int) -> None:
    conf = None if src.conf is None else src.conf[src._view_index(panel.view), t]
    _cv.draw_skeleton_2d(
        canvas,
        src.points2d(panel.view, t, panel.crop, panel.stage),
        src.skeleton,
        x0=panel.x0,
        y0=panel.y0,
        scale=panel.scales(*src.view_size(panel.view, panel.crop)),
        conf=conf,
        **panel.options,
    )


def _op_skeleton_3d(canvas: np.ndarray, panel: Panel, src: Sources, t: int) -> None:
    _cv.draw_skeleton_3d(
        canvas,
        src.points3d(t, panel.stage),
        src.camera(panel.view, panel.crop),
        src.skeleton,
        x0=panel.x0,
        y0=panel.y0,
        scale=panel.scales(*src.view_size(panel.view, panel.crop)),
        **panel.options,
    )


def _op_skeleton_nmf(canvas: np.ndarray, panel: Panel, src: Sources, t: int) -> None:
    if src.nmf_pts3d is None:
        raise ValueError("skeleton_nmf panel needs Sources.nmf_pts3d")
    # The fitted model joints are in the skeleton's point order with its bones, so
    # the 3D skeleton drawer reprojects them into the view directly.
    _cv.draw_skeleton_3d(
        canvas,
        src.nmf_pts3d[t],
        src.camera(panel.view, panel.crop),
        src.skeleton,
        x0=panel.x0,
        y0=panel.y0,
        scale=panel.scales(*src.view_size(panel.view, panel.crop)),
        **panel.options,
    )


def _op_mesh_nmf(canvas: np.ndarray, panel: Panel, src: Sources, t: int) -> None:
    if src.nmf_pts3d is None:
        raise ValueError("mesh_nmf panel needs Sources.nmf_pts3d")
    from ..inverse_kinematics.mesh import load_nmf_mesh
    from . import mesh as _mesh

    nmf = load_nmf_mesh()
    # Pose once per frame (cached on Sources): every view reuses the same vertex
    # array, so the GPU rasterizer uploads the frame's geometry only once. The
    # head/abdomen size is data-estimated by the IK stage (not a panel knob).
    # Already has the configured hidden parts (default: wings) dropped, and every panel of
    # this frame gets the SAME arrays -- which is what lets the rasterizer reuse the upload.
    verts, valid = src.nmf_posed(t)
    view_h, view_w = src.view_size(panel.view, panel.crop)
    _mesh.draw_mesh_overlay(
        canvas,
        verts,
        nmf.faces,
        nmf.face_rgb,
        valid,
        src.camera(panel.view, panel.crop),
        view_h,
        view_w,
        x0=panel.x0,
        y0=panel.y0,
        scale=panel.scales(view_h, view_w),
        **panel.options,
    )


#: ``plot`` name -> draw op. Extend to add new panel kinds.
OPS: dict[str, Callable[[np.ndarray, Panel, Sources, int], None]] = {
    "imshow": _op_imshow,
    "skeleton_2d": _op_skeleton_2d,
    "skeleton_3d": _op_skeleton_3d,
    "skeleton_nmf": _op_skeleton_nmf,
    "mesh_nmf": _op_mesh_nmf,
}


# -- config parsing -----------------------------------------------------------


def _op_kwargs(table: dict, plot: str) -> dict:
    """Look up a per-op kwargs table's entry for ``plot`` (a dict, else empty)."""
    value = table.get(plot, {})
    if not isinstance(value, dict):
        raise ValueError(
            f"visualization kwargs for plot op {plot!r} must be a table of keyword "
            f"arguments, got {value!r}"
        )
    return value


#: Grid cells that mean "leave this tile empty".
_GRID_GAP = frozenset({"", "-", "."})


def _expand_grid(entry: dict, viz: dict, config: "Config", loc: str) -> list[dict]:
    """Expand a video's ``grid`` into ``imshow`` + overlay panels with computed offsets.

    A montage is the overwhelmingly common video and it is pure repetition: every panel
    is the same two ops at a position that is a row and a column times the cell size. Two
    of those three facts were being written by hand, per panel, per video -- fourteen
    lines each to say "three by three", with the pixel offsets recomputed by a person
    whenever a cell size changed. So a video may say the shape instead::

        [[visualization.videos]]
        video_name = "pose3d"
        plot  = "skeleton_3d"
        stage = "triangulation"
        grid  = [["rf", "f",    "lf"],
                 ["rm", "bird", "lm"],
                 ["rh", "h",    "lh"]]

    Cell size comes from the resolved ``imshow`` width/height (the same
    ``[visualization.kwargs]`` the panels already read), or from an explicit
    ``cell = [w, h]``. A cell of ``""``, ``"-"`` or ``"."`` leaves its tile empty.

    A cell whose view is **not a camera** gets no ``imshow`` -- the synthetic ``bird``
    plan view has no footage to draw under its skeleton, and with ``plot = "skeleton_2d"``
    it has nothing to draw at all, so it is skipped entirely rather than drawn onto black.
    Deriving that from ``[cameras.*]`` rather than asking is the whole point: a rig that
    gains a camera gains a drawable cell without anyone editing a second list.

    Explicit ``panels`` are still honored and are appended after the expansion, so a
    one-off tile can be added to a grid without abandoning it.

    Returns
    -------
    list of dict
        Panel tables in the same shape a hand-written ``panels`` list produces.
    """
    grid = entry.get("grid")
    if grid is None:
        return []
    if not isinstance(grid, list) or not all(isinstance(r, list) for r in grid):
        raise ValueError(f"{loc} 'grid' must be a list of rows (lists of view names)")
    plot = entry.get("plot")
    if plot is None:
        raise ValueError(
            f"{loc} has a 'grid' but no 'plot' -- name the overlay op drawn on each "
            f"cell (one of {sorted(OPS)})"
        )
    if plot not in OPS:
        raise ValueError(
            f"{loc} has unknown plot op {plot!r}; choose from {sorted(OPS)}"
        )

    cell = entry.get("cell")
    if cell is None:
        merged = {
            **_op_kwargs(viz.get("kwargs", {}), "imshow"),
            **_op_kwargs(entry.get("kwargs", {}), "imshow"),
        }
        cell = [merged.get("width"), merged.get("height")]
    if len(cell) != 2 or not all(isinstance(v, int) and v > 0 for v in cell):
        raise ValueError(
            f"{loc} needs a cell size for its 'grid': set cell = [width, height] on the "
            "video, or give [visualization.kwargs] imshow both a width and a height"
        )
    cw, ch = int(cell[0]), int(cell[1])

    cameras = set(config.camera_table()[1])
    extras = {k: entry[k] for k in ("stage",) if k in entry}
    panels: list[dict] = []
    for r, row in enumerate(grid):
        for c, view in enumerate(row):
            if not isinstance(view, str):
                raise ValueError(
                    f"{loc} grid[{r}][{c}] must be a view name, got {view!r}"
                )
            if view in _GRID_GAP:
                continue
            at = {"view": view, "x0": c * cw, "y0": r * ch}
            if view in cameras:
                panels.append({"plot": "imshow", **at})
            elif plot == "skeleton_2d":
                continue  # no footage, so no 2D detections to draw on it
            panels.append({"plot": plot, **at, **extras})
    return panels


def _layout_key(panel: dict, options: dict, key: str):
    """Resolve a structural layout key (``scale`` / ``width`` / ``height``).

    A direct key on the panel wins over one merged in from the op-kwargs levels.
    The key is popped from ``options`` either way so it is never forwarded to the
    draw op (it resizes the layer, it is not a draw argument).

    Parameters
    ----------
    panel
        The raw panel table.
    options
        The merged op-kwargs; ``key`` is popped from it.
    key
        The layout key (``"scale"`` / ``"width"`` / ``"height"``).

    Returns
    -------
    The resolved value, or ``None`` if unset at every level.
    """
    value = panel[key] if key in panel else options.get(key)
    options.pop(key, None)
    return value


def _parse_crop(value, loc: str) -> tuple[int, int, int, int] | str | None:
    """Validate a panel ``crop``: ``(x, y, w, h)`` positive-size ints, or a reference.

    A string is returned verbatim as an unresolved reference (see
    :func:`_referenced_transforms`); a box is checked here rather than at draw time
    because a malformed window fails as a silently empty or transposed slice -- a black
    panel or a stretched one -- and the config is where a person can still see what they
    meant.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    box = tuple(value)
    if len(box) != 4:
        raise ValueError(f"{loc} crop must be [x, y, width, height], got {value!r}")
    x, y, w, h = (int(v) for v in box)
    if w <= 0 or h <= 0:
        raise ValueError(f"{loc} crop has non-positive size: {value!r}")
    if x < 0 or y < 0:
        raise ValueError(f"{loc} crop starts outside the frame: {value!r}")
    return x, y, w, h


def _referenced_transforms(
    ref: str, view: str, plan, loc: str
) -> tuple["FrameTransform", ...]:
    """The preprocessing chain(s) a string ``crop`` refers to (empty = the whole frame).

    ``"pose2d"`` (:data:`DETECTOR_CROP`) resolves per view: whatever the pathways feeding
    this view detect through, so one setting under ``[visualization]`` gives each panel
    its own camera's box. Any other string names a ``[[pose2d.preprocessors]]`` entry
    directly, for the cases the per-view rule cannot answer -- a panel whose view the
    detector does not run on, or one of two pathways that window a view differently.

    Empty when the view has nothing to borrow: a view no pathway writes (the derived
    ``"bird"`` plan view, or a camera the detector skips) is left uncropped rather than
    treated as an error, which is what makes a single global setting safe to write.

    Parameters
    ----------
    ref
        The reference string.
    view
        The panel's view (only ``"pose2d"`` uses it).
    plan
        The config's :class:`~deeperfly.pose2d.pathways.DetectionPlan`.
    loc
        A config location for error messages.

    Returns
    -------
    tuple of FrameTransform
        The candidate chains, resolved to a window later (:meth:`Sources.window`).
    """
    if ref == DETECTOR_CROP:
        if ref in plan.preprocessors:
            raise ValueError(
                f"{loc} crop = {ref!r} means the window this view's [[pose2d.pathways]] "
                f"detect through, so a [[pose2d.preprocessors]] cannot be named {ref!r}; "
                f"rename the preprocessor."
            )
        return plan.view_transforms().get(view, ())
    try:
        return (plan.preprocessors[ref],)
    except KeyError:
        raise ValueError(
            f"{loc} crop = {ref!r} is neither {DETECTOR_CROP!r} (the window this view's "
            f"pathways detect through) nor a [[pose2d.preprocessors]] name "
            f"(have: {sorted(plan.preprocessors) or 'none'}); write an explicit "
            f"[x, y, width, height] box to crop to something the detector has no opinion "
            f"about."
        ) from None


def _require_keys(table: dict, required: tuple[str, ...], loc: str) -> None:
    """Raise a clear ``ValueError`` if ``table`` lacks any ``required`` key.

    Gives ``[[visualization.videos]]`` the same typo-friendly errors the
    static config sections get (see :func:`deeperfly.config._params`) instead of a
    raw ``KeyError`` surfacing from deep in the parser.
    """
    missing = [k for k in required if k not in table]
    if missing:
        raise ValueError(f"{loc} is missing required key(s) {missing}")


class _PlanOnDemand:
    """The config's detection plan, built at most once and only if a panel asks.

    Reading the videos must keep working for a config that has no detector at all -- a
    library caller compositing panels over frames it brought itself -- so the plan is
    built lazily, and a config that cannot produce one only fails for the panel that
    wanted it, naming the crop that reached for it.
    """

    def __init__(self, config: "Config") -> None:
        self._config = config
        self._plan = None

    def get(self, loc: str):
        if self._plan is None:
            try:
                self._plan = self._config.detection_plan()
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"{loc} crop refers to the 2D detection plan, which this config "
                    f"cannot build ({exc}); write the crop as an explicit "
                    f"[x, y, width, height] box instead."
                ) from exc
        return self._plan


def read_video_specs(config: "Config") -> list[VideoSpec]:
    """Parse ``[[visualization.videos]]`` from a config.

    A video's panels come from its ``grid`` (see :func:`_expand_grid`), its explicit
    ``panels`` list, or both -- the grid expands first and the explicit panels follow, so
    they draw on top. A video with neither draws nothing and is rejected.

    Per-op kwargs are merged into each panel's ``options`` from least to most
    specific: global ``[visualization.kwargs]``, the video entry's
    ``kwargs``, then the panel's own extra keys (each keyed by ``plot`` op name).
    The layout keys ``scale`` / ``width`` / ``height`` are lifted onto the
    :class:`Panel` fields rather than forwarded. The canvas background comes from
    ``visualization.background`` (default ``"black"``).

    ``crop`` resolves across the same three levels (``[visualization]`` -> the video
    entry -> the panel), so ``crop = "pose2d"`` written once gives every panel the window
    its own view's detector looked through. A referenced crop becomes
    :attr:`Panel.crop_from` and is turned into a box at render time, when the raw frame
    size is known.

    Parameters
    ----------
    config
        A :class:`~deeperfly.config.Config`.

    Returns
    -------
    list of VideoSpec
        One spec per ``[[visualization.videos]]`` entry.
    """
    viz = config.visualization
    global_kwargs = viz.get("kwargs", {})
    background = viz.get("background", "black")
    global_fps = (viz.get("output_fps"), viz.get("speed"))
    plan = _PlanOnDemand(config)  # only built if some panel references it
    specs: list[VideoSpec] = []
    for i, entry in enumerate(viz.get("videos", [])):
        loc = f"[[visualization.videos]] (entry {i})"
        _require_keys(entry, ("video_name",), loc)
        video_kwargs = entry.get("kwargs", {})
        panels = []
        raw_panels = [*_expand_grid(entry, viz, config, loc), *entry.get("panels", [])]
        for j, p in enumerate(raw_panels):
            ploc = f"{loc} panel {j}"
            _require_keys(p, ("plot", "view"), ploc)
            plot = p["plot"]
            if plot not in OPS:
                raise ValueError(
                    f"{ploc} has unknown plot op {plot!r}; choose from {sorted(OPS)}"
                )
            options = {
                **_op_kwargs(global_kwargs, plot),
                **_op_kwargs(video_kwargs, plot),
                **{k: v for k, v in p.items() if k not in _RESERVED},
            }
            # scale / width / height resize the layer rather than reaching the draw
            # op, so pull them out of the merged kwargs.
            scale = _layout_key(p, options, "scale")
            width = _layout_key(p, options, "width")
            height = _layout_key(p, options, "height")
            # crop: panel -> video -> global, most specific winning.
            crop = _parse_crop(p.get("crop", entry.get("crop", viz.get("crop"))), ploc)
            crop_from: tuple["FrameTransform", ...] = ()
            if isinstance(crop, str):
                # A borrowed chain is a WINDOW and nothing else now that the op grammar
                # is gone, so there is no mirrored-or-turned case left to warn about.
                crop_from = _referenced_transforms(
                    crop, p["view"], plan.get(ploc), ploc
                )
                crop = None
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
                    crop=crop,
                    clip=bool(p.get("clip", True)),
                    stage=p.get("stage"),
                    crop_from=crop_from,
                    options=options,
                )
            )
        output_fps, speed = _resolve_fps_spec(entry, *global_fps)
        specs.append(
            VideoSpec(
                video_name=entry["video_name"],
                panels=panels,
                width=entry.get("width"),
                height=entry.get("height"),
                background=background,
                output_fps=output_fps,
                speed=speed,
            )
        )
    return specs


def _resolve_fps_spec(
    entry: dict, global_output_fps, global_speed
) -> tuple[float | None, float | None]:
    """Pick this video's ``(output_fps, speed)``, most specific level winning.

    A per-video ``output_fps`` or ``speed`` overrides the global
    ``[visualization]`` setting, and within one level an explicit
    ``output_fps`` beats ``speed``. Exactly one of the pair is set (or both
    ``None``), so :meth:`VideoSpec.resolve_fps` never has to break a tie.

    Parameters
    ----------
    entry
        One video entry table.
    global_output_fps, global_speed
        The ``[visualization]`` fallback values.

    Returns
    -------
    output_fps, speed : float or None
        Exactly one set (or both ``None``).
    """
    if entry.get("output_fps") is not None:
        return float(entry["output_fps"]), None
    if entry.get("speed") is not None:
        return None, float(entry["speed"])
    if global_output_fps is not None:
        return float(global_output_fps), None
    if global_speed is not None:
        return None, float(global_speed)
    return None, None


# -- rendering ----------------------------------------------------------------


def canvas_size(spec: VideoSpec, src: Sources) -> tuple[int, int]:
    """``(height, width)`` for ``spec``: explicit when set, else panel bbox.

    Parameters
    ----------
    spec
        The video spec.
    src
        The data sources (for the per-view sizes).

    Returns
    -------
    height, width : int
        The canvas size in pixels.
    """
    height, width = spec.height, spec.width
    if height is None or width is None:
        bbox_h = bbox_w = 0
        for panel in spec.panels:
            panel = panel.resolve_crop(src)
            ph, pw = panel.footprint(*src.view_size(panel.view, panel.crop))
            bbox_h = max(bbox_h, panel.y0 + ph)
            bbox_w = max(bbox_w, panel.x0 + pw)
        height = bbox_h if height is None else height
        width = bbox_w if width is None else width
    return int(height), int(width)


def compose_frame(spec: VideoSpec, src: Sources, t: int) -> np.ndarray:
    """Composite frame ``t`` of ``spec`` into a single RGB array.

    Parameters
    ----------
    spec
        The video spec (its ordered panels).
    src
        The data sources the panels draw from.
    t
        The frame index.

    Returns
    -------
    np.ndarray
        The composited ``(H, W, 3)`` uint8 RGB frame.

    Raises
    ------
    ValueError
        If a panel names an unknown ``plot`` op.
    """
    height, width = canvas_size(spec, src)
    canvas = _cv.new_canvas(height, width, spec.background)
    for panel in spec.panels:
        try:
            op = OPS[panel.plot]
        except KeyError:
            raise ValueError(
                f"unknown plot op {panel.plot!r}; choose from {sorted(OPS)}"
            ) from None
        # Every draw op reads panel.crop as a concrete box, so a borrowed one becomes one
        # here -- before the footprint is measured, since the window sizes the panel.
        panel = panel.resolve_crop(src)
        ph, pw = panel.footprint(*src.view_size(panel.view, panel.crop))
        if panel.background is not None:
            _cv.fill_region(canvas, panel.x0, panel.y0, pw, ph, panel.background)
        target, placed = _panel_target(canvas, panel, ph, pw)
        if target is None:
            continue  # the footprint is entirely off-canvas
        op(target, placed, src, t)
    return canvas


def _panel_target(
    canvas: np.ndarray, panel: Panel, ph: int, pw: int
) -> tuple[np.ndarray | None, Panel]:
    """The array a panel draws into, and the panel repositioned for it.

    Clipping is done by handing the op a **sliced view** of the canvas rather than by
    compositing into a private tile and blitting it back. That matters: panels layer --
    the montage draws ``imshow`` and then ``skeleton_*`` at the same offset -- and an
    opaque tile would erase whatever the previous panel put there. A numpy slice is a
    writable window with a larger row stride, which is exactly an OpenCV ROI, so the
    draw ops keep working and simply have nowhere to spill.

    Panels placed at a negative offset are drawn unclipped: a slice cannot express the
    part hanging off the top-left, and a panel deliberately positioned off-canvas is
    asking to be cropped by the canvas rather than by its own footprint.
    """
    if not panel.clip or panel.x0 < 0 or panel.y0 < 0:
        return canvas, panel
    h, w = canvas.shape[:2]
    y1, x1 = min(panel.y0 + ph, h), min(panel.x0 + pw, w)
    if y1 <= panel.y0 or x1 <= panel.x0:
        return None, panel
    from dataclasses import replace as _replace

    return canvas[panel.y0 : y1, panel.x0 : x1], _replace(panel, x0=0, y0=0)


#: Draw ops that are NOT thread-safe (a shared GPU/OpenGL rasterizer context), so a
#: spec using one is composited serially even when ``workers > 1``. The OpenCV ops
#: (imshow, skeleton_*) release the GIL and composite fine in parallel.
_GL_OPS = frozenset({"mesh_nmf"})


def _spec_is_thread_safe(spec: VideoSpec) -> bool:
    return not any(p.plot in _GL_OPS for p in spec.panels)


def _composited_in_order(spec: VideoSpec, src: Sources, n: int, workers: int):
    """Yield ``compose_frame(spec, src, t)`` for ``t`` in ``0..n`` **in order**, with
    up to ``workers`` frames composited concurrently.

    ``compose_frame`` builds its own canvas and only reads ``src``, so frames are
    independent; the OpenCV draw ops release the GIL, so a thread pool overlaps
    compositing across cores (and with the consumer's encode). A bounded
    look-ahead deque preserves output order and caps peak memory at a few frames.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    lookahead = max(2, workers * 2)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque = deque()
        it = iter(range(n))
        for _ in range(lookahead):  # prime the pipeline
            t = next(it, None)
            if t is None:
                break
            pending.append(pool.submit(compose_frame, spec, src, t))
        while pending:
            fut = pending.popleft()
            t = next(it, None)
            if t is not None:  # keep the pipeline full as each frame is drained
                pending.append(pool.submit(compose_frame, spec, src, t))
            yield fut.result()


def stream_video(
    spec: VideoSpec,
    src: Sources,
    *,
    n_frames: int | None = None,
    progress: Callable[[Iterable[int]], Iterable[int]] | None = None,
    workers: int = 1,
) -> Iterator[Float[np.ndarray, "H W 3"]]:
    """Composite ``spec`` frame by frame, yielding each ``(H, W, 3)`` uint8 frame.

    The streaming counterpart of :func:`render_video`: a consumer (e.g.
    :class:`deeperfly.io.VideoWriter`) encodes each frame as it is composited, so a
    long clip never has to be held in memory as one ``(T, H, W, 3)`` array.

    Parameters
    ----------
    spec
        The video spec.
    src
        The data sources the panels draw from.
    n_frames
        How many frames to render (defaults to ``src.n_frames()``).
    progress
        Optional wrapper of the per-frame iterator (e.g. a rich progress bar);
        defaults to the identity, keeping this library UI-free.
    workers
        Compositing threads. ``> 1`` composites frames concurrently (still yielded
        in order) -- a large speedup for the OpenCV panels. Ignored (forced to 1)
        for a spec with a thread-unsafe GPU op (see :data:`_GL_OPS`).

    Yields
    ------
    np.ndarray
        Each composited ``(H, W, 3)`` uint8 RGB frame, in order.
    """
    n = src.n_frames() if n_frames is None else n_frames
    if workers > 1 and _spec_is_thread_safe(spec):
        frames: Iterable = _composited_in_order(spec, src, n, workers)
    else:
        frames = (compose_frame(spec, src, t) for t in range(n))
    yield from (frames if progress is None else progress(frames))


def render_video(
    spec: VideoSpec,
    src: Sources,
    *,
    n_frames: int | None = None,
    progress: Callable[[Iterable[int]], Iterable[int]] | None = None,
) -> Float[np.ndarray, "T H W 3"]:
    """Composite every frame of ``spec`` into a ``(T, H, W, 3)`` uint8 stack.

    The eager counterpart of :func:`stream_video`, for callers that want the whole
    clip as one array; stream the frames instead when memory matters.

    Parameters
    ----------
    spec
        The video spec.
    src
        The data sources the panels draw from.
    n_frames
        How many frames to render (defaults to ``src.n_frames()``).
    progress
        Optional wrapper of the per-frame iterator (e.g. a rich progress bar);
        defaults to the identity, keeping this library UI-free.

    Returns
    -------
    np.ndarray
        The composited ``(T, H, W, 3)`` uint8 stack.
    """
    return np.stack(list(stream_video(spec, src, n_frames=n_frames, progress=progress)))


def render_videos(
    config: dict | str | Path,
    src: Sources,
    outdir: str | Path,
    *,
    fps: float = 30.0,
) -> list[Path]:
    """Render every ``[[visualization.videos]]`` to ``<outdir>/<name>.mp4``.

    Parameters
    ----------
    config
        A :class:`~deeperfly.config.Config`, parsed ``dict`` or config TOML path.
    src
        The data sources the panels draw from.
    outdir
        The directory the MP4s are written to (created if missing).
    fps
        Output frame rate.

    Returns
    -------
    list of Path
        The written MP4 paths.
    """
    from ..io import VideoWriter

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for spec in read_video_specs(config):  # type: ignore[arg-type]
        path = outdir / f"{spec.video_name}.mp4"
        # Stream frame by frame into the encoder -- never hold the whole clip.
        with VideoWriter(path, fps=fps) as writer:
            writer.write_frames(stream_video(spec, src))
        paths.append(path)
    return paths
