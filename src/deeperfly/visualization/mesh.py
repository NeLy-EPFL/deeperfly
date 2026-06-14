"""Software rasterizer for the NeuroMechFly mesh overlay (shaded, translucent).

Projects a posed mesh (world vertices + faces + per-face color, from
:meth:`deeperfly.inverse_kinematics.mesh.NmfMesh.pose`) through one camera and
draws it as a flat-shaded, depth-ordered, semi-transparent surface. Used by both
the ``mesh_nmf`` video op and the GUI's on-demand mesh overlay, so the two render
identically.

Triangles are flat-shaded by their world-space normal against a headlight at the
camera, back-to-front painted (painter's algorithm) into an opaque RGB tile plus a
coverage mask, then alpha-blended onto the target so the mesh reads as a model laid
over the fly while its internal depth shading stays visible. Faces outside the tile,
behind the camera, or with a non-finite vertex are dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from jaxtyping import Float

if TYPE_CHECKING:
    from ..cameras import Camera

__all__ = ["render_mesh_rgba", "draw_mesh_overlay"]

#: Flat-shading terms: ambient floor + diffuse gain against the headlight.
_AMBIENT = 0.45
_DIFFUSE = 0.55


def render_mesh_rgba(
    verts_world: Float[np.ndarray, "Nv 3"],
    faces: np.ndarray,
    face_rgb: np.ndarray,
    valid: np.ndarray,
    camera: "Camera",
    height: int,
    width: int,
    *,
    alpha: float = 0.55,
) -> np.ndarray:
    """Rasterize a posed mesh through ``camera`` into an ``(H, W, 4)`` RGBA overlay.

    Parameters
    ----------
    verts_world
        ``(Nv, 3)`` posed world vertices (NaN allowed; such faces are dropped).
    faces
        ``(Nf, 3)`` triangle vertex indices.
    face_rgb
        ``(Nf, 3)`` uint8 base color per face.
    valid
        ``(Nf,)`` bool mask of drawable faces (all three vertices finite).
    camera
        The camera to project + depth-sort through.
    height, width
        The overlay size in pixels (the view's native resolution).
    alpha
        Opacity of the composited mesh (0 transparent .. 1 opaque).

    Returns
    -------
    np.ndarray
        ``(H, W, 4)`` uint8 RGBA; the alpha channel is ``round(alpha * coverage)``.
    """
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    cover = np.zeros((height, width), dtype=np.uint8)

    pts2d = np.asarray(camera.project(np.nan_to_num(verts_world)), dtype=float)
    rmat, tvec = np.asarray(camera.rmat), np.asarray(camera.tvec)
    depth = (verts_world @ rmat.T + tvec)[:, 2]

    tri = faces[valid]
    if tri.size == 0:
        return np.dstack([rgb, cover])
    tri_xy = pts2d[tri]  # (F, 3, 2)
    tri_z = depth[tri]  # (F, 3)
    colors = face_rgb[valid].astype(float)

    keep = (tri_z > 0).all(axis=1) & np.isfinite(tri_xy).all(axis=(1, 2))
    # Frustum cull: drop triangles whose bounding box misses the tile entirely.
    lo = tri_xy.min(axis=1)
    hi = tri_xy.max(axis=1)
    keep &= (hi[:, 0] >= 0) & (lo[:, 0] < width) & (hi[:, 1] >= 0) & (lo[:, 1] < height)
    tri_xy, tri_z, colors = tri_xy[keep], tri_z[keep], colors[keep]
    if len(tri_xy) == 0:
        return np.dstack([rgb, cover])

    shade = _shading(verts_world, faces[valid][keep], rmat, tvec)
    shaded = np.clip(colors * shade[:, None], 0, 255).astype(np.int32)

    order = np.argsort(-tri_z.mean(axis=1))  # far -> near (painter's)
    poly = np.round(tri_xy[order]).astype(np.int32)
    shaded = shaded[order]
    for i in range(len(poly)):
        c = (int(shaded[i, 2]), int(shaded[i, 1]), int(shaded[i, 0]))  # cv2 is BGR
        cv2.fillConvexPoly(rgb, poly[i], c[::-1], lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(cover, poly[i], 255, lineType=cv2.LINE_AA)

    a = (
        cover.astype(np.uint16) * int(round(np.clip(alpha, 0, 1) * 255)) // 255
    ).astype(np.uint8)
    return np.dstack([rgb, a])


def _shading(verts_world, faces, rmat, tvec) -> np.ndarray:
    """Per-face Lambert intensity from a headlight at the camera (two-sided)."""
    v0, v1, v2 = (
        verts_world[faces[:, 0]],
        verts_world[faces[:, 1]],
        verts_world[faces[:, 2]],
    )
    normal = np.cross(v1 - v0, v2 - v0)
    n = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.where(n > 1e-12, n, 1.0)
    # Light from the camera centre toward each face (a headlight); two-sided so a
    # flipped winding after mesh simplification still lights up rather than going black.
    centre = -rmat.T @ tvec
    to_light = centre - (v0 + v1 + v2) / 3.0
    to_light = to_light / np.linalg.norm(to_light, axis=1, keepdims=True).clip(1e-12)
    lambert = np.abs((normal * to_light).sum(axis=1))
    return _AMBIENT + _DIFFUSE * lambert


def draw_mesh_overlay(
    canvas: np.ndarray,
    verts_world: Float[np.ndarray, "Nv 3"],
    faces: np.ndarray,
    face_rgb: np.ndarray,
    valid: np.ndarray,
    camera: "Camera",
    view_h: int,
    view_w: int,
    *,
    x0: int = 0,
    y0: int = 0,
    scale: tuple[float, float] = (1.0, 1.0),
    alpha: float = 0.55,
) -> np.ndarray:
    """Render the posed mesh and alpha-blend it onto ``canvas`` at ``(x0, y0)``.

    Mirrors :func:`deeperfly.visualization.opencv.draw_skeleton_3d`'s placement:
    the mesh is rasterized at the view's native size, resized by ``scale``, and
    composited into the canvas tile (clipped to the canvas bounds). Returns the
    same ``canvas``.
    """
    rgba = render_mesh_rgba(
        verts_world, faces, face_rgb, valid, camera, view_h, view_w, alpha=alpha
    )
    sx, sy = scale
    if sx != 1.0 or sy != 1.0:
        out_w = max(1, int(round(view_w * sx)))
        out_h = max(1, int(round(view_h * sy)))
        rgba = cv2.resize(rgba, (out_w, out_h), interpolation=cv2.INTER_AREA)
    _alpha_blit(canvas, rgba, x0, y0)
    return canvas


def _alpha_blit(canvas: np.ndarray, rgba: np.ndarray, x0: int, y0: int) -> None:
    """Alpha-composite an RGBA tile onto ``canvas`` at ``(x0, y0)`` (clipped)."""
    ch, cw = canvas.shape[:2]
    th, tw = rgba.shape[:2]
    x1, y1 = min(x0 + tw, cw), min(y0 + th, ch)
    x0c, y0c = max(x0, 0), max(y0, 0)
    if x0c >= x1 or y0c >= y1:
        return
    tile = rgba[y0c - y0 : y1 - y0, x0c - x0 : x1 - x0]
    a = (tile[..., 3:4].astype(np.float32)) / 255.0
    region = canvas[y0c:y1, x0c:x1].astype(np.float32)
    canvas[y0c:y1, x0c:x1] = (
        region * (1 - a) + tile[..., :3].astype(np.float32) * a
    ).astype(canvas.dtype)
