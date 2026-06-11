"""OpenCV pose-overlay primitives drawn straight into RGB image buffers.

Each primitive draws onto a caller-supplied ``canvas`` (an ``(H, W, 3)`` uint8 RGB
array) at a pixel offset ``(x0, y0)``, so a compositor can layer several -- an
image, then a skeleton on top -- into one frame (see
:mod:`deeperfly.visualization.compose`). Drawing goes directly into the array with ``cv2``,
far faster than matplotlib for video.

For 3D, bones and joints are ordered back-to-front by camera-space depth (the
painter's algorithm), so nearer limbs occlude farther ones; points behind the
camera are dropped. Buffers are RGB throughout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from jaxtyping import Float

from ._palette import point_colors_rgb

if TYPE_CHECKING:  # avoid importing the camera/skeleton modules at drawing time
    from ..cameras import Camera
    from ..skeleton import Skeleton

__all__ = [
    "new_canvas",
    "fill_region",
    "draw_image",
    "draw_skeleton_2d",
    "draw_skeleton_3d",
]

#: Named background colors (RGB) for :func:`new_canvas`.
BACKGROUNDS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}

Color = tuple[int, int, int]

#: A resize factor: one number (uniform) or an ``(sx, sy)`` pair (per axis).
Scale = float | tuple[float, float]


def _xy_scale(scale: Scale) -> tuple[float, float]:
    """Normalize a uniform factor or an ``(sx, sy)`` pair to ``(sx, sy)`` floats."""
    if isinstance(scale, (int, float)):
        return float(scale), float(scale)
    sx, sy = scale
    return float(sx), float(sy)


def _bg_rgb(background: str | tuple[int, int, int]) -> Color:
    if isinstance(background, str):
        try:
            return BACKGROUNDS[background]
        except KeyError:
            raise ValueError(
                f"background must be one of {sorted(BACKGROUNDS)} or an RGB "
                f"tuple; got {background!r}"
            ) from None
    r, g, b = background
    return int(r), int(g), int(b)


def new_canvas(
    height: int, width: int, background: str | tuple[int, int, int] = "black"
) -> Float[np.ndarray, "H W 3"]:
    """A fresh ``(height, width, 3)`` uint8 RGB canvas filled with ``background``.

    Parameters
    ----------
    height, width
        Canvas size in pixels.
    background
        A named color (:data:`BACKGROUNDS`) or an RGB tuple.

    Returns
    -------
    np.ndarray
        The filled ``(height, width, 3)`` uint8 RGB canvas.
    """
    canvas = np.empty((int(height), int(width), 3), dtype=np.uint8)
    canvas[:] = _bg_rgb(background)
    return canvas


def fill_region(
    canvas: np.ndarray,
    x0: int,
    y0: int,
    width: int,
    height: int,
    background: str | tuple[int, int, int],
) -> np.ndarray:
    """Paint the ``width x height`` rectangle at ``(x0, y0)`` with ``background``.

    The rectangle is clipped to ``canvas`` (a partly off-canvas region paints only
    its visible part). Use it to give one panel's footprint its own backdrop
    before an op draws on top of it.

    Parameters
    ----------
    canvas
        The RGB buffer painted in place.
    x0, y0
        Top-left pixel of the rectangle.
    width, height
        Rectangle size in pixels.
    background
        A named color (:data:`BACKGROUNDS`) or an RGB tuple.

    Returns
    -------
    np.ndarray
        The same ``canvas``.
    """
    color = _bg_rgb(background)
    h_canvas, w_canvas = canvas.shape[:2]
    xs, ys = max(int(x0), 0), max(int(y0), 0)
    xe, ye = min(int(x0) + int(width), w_canvas), min(int(y0) + int(height), h_canvas)
    if xs < xe and ys < ye:
        canvas[ys:ye, xs:xe] = color
    return canvas


def _as_rgb_u8(image: np.ndarray) -> np.ndarray:
    """Coerce an image to a contiguous ``(H, W, 3)`` uint8 RGB array."""
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if np.issubdtype(img.dtype, np.floating):
        scale = 255.0 if float(img.max(initial=0.0)) <= 1.0 + 1e-6 else 1.0
        img = np.clip(img * scale, 0, 255)
    return np.ascontiguousarray(img, dtype=np.uint8)


def draw_image(
    canvas: np.ndarray,
    image: np.ndarray,
    x0: int = 0,
    y0: int = 0,
    scale: Scale = 1.0,
) -> np.ndarray:
    """Blit ``image`` onto ``canvas`` with its top-left at ``(x0, y0)`` (clipped).

    ``scale`` resizes ``image`` first (e.g. ``0.5`` for a half-size tile); pass an
    ``(sx, sy)`` pair to resize the axes independently (e.g. to fit an exact
    pixel box).

    Parameters
    ----------
    canvas
        The RGB buffer drawn on in place.
    image
        The source image (grayscale or RGB[A], float or uint8).
    x0, y0
        Top-left pixel where the (resized) image is blitted.
    scale
        A uniform factor or an ``(sx, sy)`` pair applied before blitting.

    Returns
    -------
    np.ndarray
        The same ``canvas``.
    """
    img = _as_rgb_u8(image)
    sx, sy = _xy_scale(scale)
    if (sx, sy) != (1.0, 1.0):
        h, w = img.shape[:2]
        interp = cv2.INTER_AREA if (sx < 1.0 or sy < 1.0) else cv2.INTER_LINEAR
        size = (max(1, round(w * sx)), max(1, round(h * sy)))
        img = cv2.resize(img, size, interpolation=interp)
    h, w = img.shape[:2]
    height, width = canvas.shape[:2]
    xs, ys = max(x0, 0), max(y0, 0)
    xe, ye = min(x0 + w, width), min(y0 + h, height)
    if xs >= xe or ys >= ye:
        return canvas
    canvas[ys:ye, xs:xe] = img[ys - y0 : ye - y0, xs - x0 : xe - x0]
    return canvas


def _colors_u8(skeleton: "Skeleton", palette: dict[str, str] | None) -> np.ndarray:
    return np.clip(point_colors_rgb(skeleton, palette) * 255.0 + 0.5, 0, 255).astype(
        np.uint8
    )


def _draw_point(
    canvas: np.ndarray, center: tuple[int, int], radius: int, color: Color, alpha: float
) -> None:
    """Filled anti-aliased circle, alpha-blended over its bounding ROI."""
    if alpha >= 1.0:
        cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)
        return
    if alpha <= 0.0:
        return
    x, y = center
    r = radius + 1
    height, width = canvas.shape[:2]
    x0, y0 = max(x - r, 0), max(y - r, 0)
    x1, y1 = min(x + r + 1, width), min(y + r + 1, height)
    if x0 >= x1 or y0 >= y1:
        return
    roi = canvas[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.circle(overlay, (x - x0, y - y0), radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def _draw(
    canvas: np.ndarray,
    pts: np.ndarray,
    skeleton: "Skeleton",
    *,
    colors: np.ndarray,
    depth: np.ndarray | None,
    conf: np.ndarray | None,
    x0: int,
    y0: int,
    scale_x: float,
    scale_y: float,
    point_radius: int,
    line_thickness: int,
    draw_points: bool,
) -> np.ndarray:
    """Draw bones then joints, back-to-front when ``depth`` is given."""
    pts = np.asarray(pts, dtype=float)
    finite: np.ndarray = np.asarray(np.isfinite(pts).all(-1))

    def xy(i: int) -> tuple[int, int]:
        return int(round(pts[i, 0] * scale_x + x0)), int(
            round(pts[i, 1] * scale_y + y0)
        )

    bones = skeleton.bones
    bone_order = range(len(bones))
    if depth is not None and len(bones):
        d = np.asarray(depth, dtype=float)
        mid = np.array(
            [
                (d[a] + d[b]) / 2.0 if (finite[a] and finite[b]) else -np.inf
                for a, b in bones
            ]
        )
        bone_order = np.argsort(-mid)  # farthest (largest z) first
    for k in bone_order:
        a, b = int(bones[k][0]), int(bones[k][1])
        if finite[a] and finite[b]:
            color: Color = tuple(map(int, colors[a]))  # type: ignore[assignment]
            cv2.line(canvas, xy(a), xy(b), color, line_thickness, cv2.LINE_AA)

    if not draw_points:
        return canvas
    joint_order = range(skeleton.n_points)
    if depth is not None:
        joint_order = np.argsort(-np.where(finite, np.asarray(depth, float), -np.inf))
    for n in joint_order:
        if not finite[n]:
            continue
        alpha = 1.0 if conf is None else float(np.clip(conf[n], 0.0, 1.0))
        _draw_point(canvas, xy(n), point_radius, tuple(map(int, colors[n])), alpha)  # type: ignore[arg-type]
    return canvas


def draw_skeleton_2d(
    canvas: np.ndarray,
    pts2d: Float[np.ndarray, "P 2"],
    skeleton: "Skeleton",
    *,
    x0: int = 0,
    y0: int = 0,
    scale: Scale = 1.0,
    conf: Float[np.ndarray, "P"] | None = None,
    palette: dict[str, str] | None = None,
    point_radius: int = 3,
    line_thickness: int = 1,
    draw_points: bool = True,
) -> np.ndarray:
    """Draw a single view's 2D joints + bones onto ``canvas`` at ``(x0, y0)``.

    NaN joints (and their bones) are skipped; there is no depth ordering.

    Parameters
    ----------
    canvas
        The RGB buffer drawn on in place.
    pts2d
        The view's 2D joints of shape ``(P, 2)`` in image pixels.
    skeleton
        Skeleton supplying the bones and per-limb colors.
    x0, y0
        Top-left pixel offset.
    scale
        A uniform factor or ``(sx, sy)`` pair multiplying the pixel coordinates
        (match an ``imshow`` of the same view).
    conf
        Per-joint confidence ``(P,)`` modulating opacity, or ``None``.
    palette
        Optional ``limb_name -> hex`` override of the skeleton palette.
    point_radius, line_thickness
        Joint and bone sizes in pixels.
    draw_points
        Whether to draw joints (bones are always drawn).

    Returns
    -------
    np.ndarray
        The same ``canvas``.
    """
    sx, sy = _xy_scale(scale)
    return _draw(
        canvas,
        pts2d,
        skeleton,
        colors=_colors_u8(skeleton, palette),
        depth=None,
        conf=conf,
        x0=x0,
        y0=y0,
        scale_x=sx,
        scale_y=sy,
        point_radius=point_radius,
        line_thickness=line_thickness,
        draw_points=draw_points,
    )


def draw_skeleton_3d(
    canvas: np.ndarray,
    pts3d: Float[np.ndarray, "P 3"],
    camera: "Camera",
    skeleton: "Skeleton",
    *,
    x0: int = 0,
    y0: int = 0,
    scale: Scale = 1.0,
    conf: Float[np.ndarray, "P"] | None = None,
    palette: dict[str, str] | None = None,
    point_radius: int = 3,
    line_thickness: int = 1,
    draw_points: bool = True,
) -> np.ndarray:
    """Reproject a 3D skeleton into ``camera`` and draw it onto ``canvas``.

    Bones and joints are depth-ordered back-to-front; points behind the camera are
    dropped.

    Parameters
    ----------
    canvas
        The RGB buffer drawn on in place.
    pts3d
        World-coordinate joints of shape ``(P, 3)``.
    camera
        The camera the skeleton is reprojected through (distortion included).
    skeleton
        Skeleton supplying the bones and per-limb colors.
    x0, y0
        Top-left pixel offset.
    scale
        A uniform factor or ``(sx, sy)`` pair multiplying the projected pixels
        (match an ``imshow`` of the same view).
    conf
        Per-joint confidence ``(P,)`` modulating opacity, or ``None``.
    palette
        Optional ``limb_name -> hex`` override of the skeleton palette.
    point_radius, line_thickness
        Joint and bone sizes in pixels.
    draw_points
        Whether to draw joints (bones are always drawn).

    Returns
    -------
    np.ndarray
        The same ``canvas``.
    """
    pts3d = np.asarray(pts3d, dtype=float)
    pts2d = np.asarray(camera.project(pts3d), dtype=float).copy()
    p_cam = pts3d @ np.asarray(camera.rmat).T + np.asarray(camera.tvec)
    depth = p_cam[:, 2]
    pts2d[depth <= 0] = np.nan  # behind the camera -> not drawable
    sx, sy = _xy_scale(scale)
    return _draw(
        canvas,
        pts2d,
        skeleton,
        colors=_colors_u8(skeleton, palette),
        depth=depth,
        conf=conf,
        x0=x0,
        y0=y0,
        scale_x=sx,
        scale_y=sy,
        point_radius=point_radius,
        line_thickness=line_thickness,
        draw_points=draw_points,
    )
