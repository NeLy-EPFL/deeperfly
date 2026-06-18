"""GPU rasterizer for the NeuroMechFly mesh overlay (headless OpenGL via moderngl).

The pure-numpy/OpenCV rasterizer in :mod:`deeperfly.visualization.mesh` paints
~100k triangles in a Python loop, which is too slow for interactive scrubbing and
video rendering. This module renders the same posed mesh on the GPU with a tiny
shader and a real depth buffer (so occlusion is exact rather than a painter's-order
approximation).

Unlike a naive port, almost nothing happens on the CPU per view: the posed world
vertices, their smooth per-vertex normals, and an index buffer over the drawable
faces are uploaded **once per frame**, and each camera is rendered by changing only
the model-view-projection uniform (the same trick MuJoCo's renderer uses -- upload
the geometry once, re-render per camera). The vertex shader projects with the exact
pinhole matrix :meth:`deeperfly.cameras.CameraGroup.project` uses, and the fragment
shader smooth-shades against a headlight at the camera, so the overlay is both fast
and free of the flat-shaded faceting the software path shows.

This drops lens distortion (the projection is the linear pinhole), so a camera with
non-zero distortion coefficients falls back to the software rasterizer, which keeps
distortion. OpenGL contexts are thread-affine and not safe to share, while the GUI
serves frames from a worker threadpool; so all GL work runs on a single dedicated
thread that owns one offscreen EGL context. If no context can be created (no EGL,
headless box without a GPU driver, moderngl missing) the public entry point returns
``None`` and callers fall back to the software rasterizer.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from .mesh import vertex_normals

if TYPE_CHECKING:
    from ..cameras import Camera

__all__ = ["render_mesh_rgba_gl", "gl_available"]

log = logging.getLogger("deeperfly")

#: Fallback clip planes (world units). The renderer normally fits the near/far planes
#: to the posed mesh per camera (see ``_Renderer._depth_range``) so the depth buffer's
#: precision stays on the fly -- a fixed wide range z-fights coincident surfaces (the
#: eye on the head) from the far rig cameras. These remain the defaults for a bare
#: ``_mvp`` call (e.g. in tests).
_NEAR, _FAR = 0.01, 1000.0

#: Smooth-shading terms (match the software path's _AMBIENT / _DIFFUSE).
_AMBIENT, _DIFFUSE = 0.45, 0.55

_VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;   // world position
in vec3 in_col;   // per-vertex base color, 0..1
in vec3 in_nrm;   // world-space smooth normal
out vec3 v_world;
out vec3 v_col;
out vec3 v_nrm;
void main() {
    v_world = in_pos;
    v_col = in_col;
    v_nrm = in_nrm;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

_FRAGMENT_SHADER = """
#version 330
uniform vec3 light;       // camera centre, world frame (a headlight)
uniform float ambient;
uniform float diffuse;
in vec3 v_world;
in vec3 v_col;
in vec3 v_nrm;
out vec4 f_col;
void main() {
    // Smooth (interpolated) normal; two-sided so a flipped winding still lights up.
    // A zero-length normal (degenerate / doubled geometry, e.g. at the abdomen tip,
    // where vertex_normals cancels to 0) would make normalize() NaN and paint the
    // fragment white/black -- fall back to the unlit ambient term there.
    float nlen = length(v_nrm);
    vec3 n = nlen > 1e-6 ? v_nrm / nlen : vec3(0.0);
    vec3 l = normalize(light - v_world);
    float shade = ambient + diffuse * abs(dot(n, l));
    f_col = vec4(v_col * shade, 1.0);
}
"""


def _bounding_sphere(pos: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre + radius of the drawable vertices (the sphere around their AABB).

    Only the indexed (drawable) vertices are considered, so un-posed/NaN vertices never
    enter the bounds; used to fit the depth-buffer planes to the mesh per camera.
    """
    if idx.shape[0] == 0:
        return np.zeros(3), 1.0
    sub = pos[idx]
    lo = sub.min(axis=0)
    hi = sub.max(axis=0)
    centre = ((lo + hi) * 0.5).astype(float)
    radius = float(np.linalg.norm(hi - lo)) * 0.5
    return centre, max(radius, 1e-3)


def _mvp(
    intr, rmat, tvec, w: int, h: int, near: float = _NEAR, far: float = _FAR
) -> np.ndarray:
    """The pinhole model-view-projection ``CameraGroup.project`` implies (row-major).

    View flips deeperfly's (x-right, y-down, z-forward) camera frame into OpenGL's
    (x-right, y-up, z-back) with ``D = diag(1, -1, -1)``; the projection is built
    straight from the intrinsics ``[fx, fy, cx, cy]`` and the footage size, so the
    rasterised pixels land exactly where ``project`` puts them (validated to 0 px on
    undistorted cameras). ``near``/``far`` set only the depth mapping (the third/fourth
    rows); the projected x/y are independent of them.
    """
    fx, fy, cx, cy = (float(v) for v in np.asarray(intr).reshape(-1)[:4])
    r = np.asarray(rmat, dtype=float).reshape(3, 3)
    t = np.asarray(tvec, dtype=float).reshape(3)
    d = np.diag([1.0, -1.0, -1.0])
    view = np.eye(4)
    view[:3, :3] = d @ r
    view[:3, 3] = d @ t
    a = -(far + near) / (far - near)
    b = -2.0 * far * near / (far - near)
    proj = np.array(
        [
            [2 * fx / w, 0, (w - 2 * cx) / w, 0],
            [0, 2 * fy / h, (2 * cy - h) / h, 0],
            [0, 0, a, b],
            [0, 0, -1, 0],
        ],
        dtype=float,
    )
    return proj @ view


class _Renderer:
    """Owns one offscreen EGL context and the cached GL resources for it.

    All methods run on the single worker thread that created the context (see
    :class:`_GLThread`); they must not be called directly from other threads. The
    per-vertex color is uploaded once for a given mesh; the posed vertices, normals
    and face index are re-uploaded only when the frame's vertex array changes (so
    rendering all of a frame's cameras costs one upload, then a draw each).
    """

    def __init__(self) -> None:
        import moderngl

        self.mgl = moderngl
        self.ctx = moderngl.create_context(standalone=True, backend="egl")
        self.prog = self.ctx.program(
            vertex_shader=_VERTEX_SHADER, fragment_shader=_FRAGMENT_SHADER
        )
        self.prog["ambient"].value = _AMBIENT
        self.prog["diffuse"].value = _DIFFUSE
        self.samples = min(4, int(getattr(self.ctx, "max_samples", 0)))
        self._fbos: dict[tuple[int, int], tuple] = {}
        self._pos = self.ctx.buffer(reserve=1 << 16, dynamic=True)
        self._nrm = self.ctx.buffer(reserve=1 << 16, dynamic=True)
        self._col = self.ctx.buffer(reserve=1 << 16, dynamic=True)
        self._idx = self.ctx.buffer(reserve=1 << 16, dynamic=True)
        self._vao = None
        self._col_key: object = None
        self._frame_key: object = None
        self._index_count = 0
        # Bounding sphere of the current frame's drawable mesh (world frame); the depth
        # planes are fit to it per camera so far views keep depth precision.
        self._center = np.zeros(3)
        self._radius = 1.0
        # Retain the cached arrays so their id() cannot be recycled by CPython while
        # they key the upload cache (a freed array's id could otherwise be reused by
        # a different array and falsely hit the cache).
        self._col_ref: object = None
        self._frame_refs: tuple = ()

    def _framebuffers(self, w: int, h: int):
        """The (multisample draw fbo, resolve fbo) for a size, created on demand."""
        cached = self._fbos.get((w, h))
        if cached is not None:
            return cached
        ctx, s = self.ctx, self.samples
        resolve_tex = ctx.texture((w, h), 4)
        resolve = ctx.framebuffer(color_attachments=[resolve_tex])
        if s > 0:
            draw = ctx.framebuffer(
                color_attachments=[ctx.renderbuffer((w, h), 4, samples=s)],
                depth_attachment=ctx.depth_renderbuffer((w, h), samples=s),
            )
        else:  # the driver does not support multisampling: draw straight to resolve
            draw = ctx.framebuffer(
                color_attachments=[resolve_tex],
                depth_attachment=ctx.depth_renderbuffer((w, h)),
            )
        self._fbos[(w, h)] = (draw, resolve)
        return draw, resolve

    @staticmethod
    def _grow(buf, data: bytes):
        """Re-upload ``data`` into ``buf``, growing (orphaning) it if it must."""
        if len(data) > buf.size:
            buf.orphan(max(len(data), buf.size * 2))
        else:
            buf.orphan(buf.size)  # discard the old contents (avoid a GPU stall)
        buf.write(data)

    def _ensure_color(self, faces: np.ndarray, face_rgb: np.ndarray) -> None:
        """Upload the per-vertex color once per mesh (each vertex has one base color)."""
        key = (id(face_rgb), faces.shape[0])
        if key == self._col_key:
            return
        n_verts = int(faces.max()) + 1 if faces.size else 0
        vrgb = np.zeros((n_verts, 3), dtype=np.float32)
        rgb = np.asarray(face_rgb, dtype=np.float32) / 255.0
        for k in range(3):
            vrgb[faces[:, k]] = rgb
        self._grow(self._col, vrgb.tobytes())
        self._col_key = key
        self._col_ref = face_rgb
        self._vao = None  # color buffer size changed -> rebuild the vertex array

    def _ensure_frame(self, verts: np.ndarray, faces: np.ndarray, valid: np.ndarray):
        """Upload the posed verts + smooth normals + drawable-face index for a frame."""
        key = (id(verts), id(valid))
        if key == self._frame_key:
            return
        pos = np.ascontiguousarray(verts, dtype="f4")
        nrm = vertex_normals(verts, faces, valid).astype("f4")
        idx = np.ascontiguousarray(faces[valid].reshape(-1), dtype="<u4")
        self._grow(self._pos, pos.tobytes())
        self._grow(self._nrm, nrm.tobytes())
        self._grow(self._idx, idx.tobytes())
        self._index_count = int(idx.shape[0])
        self._center, self._radius = _bounding_sphere(pos, idx)
        self._frame_key = key
        self._frame_refs = (verts, valid)

    def _depth_range(self, rmat, tvec) -> tuple[float, float]:
        """Near/far planes bracketing the mesh along this camera's optical axis.

        Fitting the planes to the posed mesh's bounding sphere (instead of a fixed wide
        range) spends the depth buffer's precision on the fly itself, so coincident
        surfaces (the eye on the head) don't z-fight when viewed from the far rig cameras.
        """
        r2 = np.asarray(rmat, dtype=float).reshape(3, 3)[2]
        t2 = float(np.asarray(tvec, dtype=float).reshape(3)[2])
        d = float(r2 @ self._center) + t2  # optical-axis distance to the sphere centre
        near = max(d - self._radius, (d + self._radius) * 1e-3, 1e-4)
        far = max(d + self._radius, near * 1.001)
        return near, far

    def _vertex_array(self):
        if self._vao is None:
            self._vao = self.ctx.vertex_array(
                self.prog,
                [
                    (self._pos, "3f", "in_pos"),
                    (self._col, "3f", "in_col"),
                    (self._nrm, "3f", "in_nrm"),
                ],
                index_buffer=self._idx,
                index_element_size=4,
            )
        return self._vao

    def render(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        face_rgb: np.ndarray,
        valid: np.ndarray,
        intr,
        rmat,
        tvec,
        w: int,
        h: int,
    ):
        """Rasterize the posed mesh through one camera to an ``(H, W, 4)`` uint8 RGBA."""
        mgl = self.mgl
        self._ensure_color(faces, face_rgb)
        self._ensure_frame(verts, faces, valid)
        if self._index_count == 0:
            return np.zeros((h, w, 4), dtype=np.uint8)
        vao = self._vertex_array()
        near, far = self._depth_range(rmat, tvec)
        mvp = _mvp(intr, rmat, tvec, w, h, near, far)
        self.prog["mvp"].write(mvp.T.astype("f4").tobytes())
        self.prog["light"].value = tuple(
            float(v) for v in -np.asarray(rmat).reshape(3, 3).T @ np.asarray(tvec)
        )
        draw, resolve = self._framebuffers(w, h)
        draw.use()
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.disable(mgl.CULL_FACE)
        draw.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        vao.render(mode=mgl.TRIANGLES, vertices=self._index_count)
        if draw is not resolve:
            self.ctx.copy_framebuffer(resolve, draw)
        raw = np.frombuffer(resolve.read(components=4, dtype="f1"), dtype=np.uint8)
        return np.flipud(raw.reshape(h, w, 4)).copy()


class _GLThread:
    """A single worker thread that lazily creates and then owns the GL renderer.

    The first task constructs the context on this thread; if that fails the thread
    records the GL backend as unavailable and every later task short-circuits.
    """

    def __init__(self) -> None:
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="deeperfly-gl"
        )
        self._renderer: _Renderer | None = None
        self._tried = False
        self._ok = False
        self._lock = threading.Lock()

    def _ensure(self) -> bool:
        if not self._tried:
            self._tried = True
            try:
                self._renderer = _Renderer()
                self._ok = True
            except Exception as exc:  # no EGL / no moderngl / no driver
                log.info("mesh GPU overlay unavailable, using CPU rasterizer (%s)", exc)
                self._ok = False
        return self._ok

    def submit(self, fn):
        """Run ``fn(renderer)`` on the GL thread; return ``None`` if GL is down."""
        with self._lock:
            fut = self._pool.submit(self._run, fn)
        return fut.result()

    def _run(self, fn):
        if not self._ensure():
            return None
        try:
            return fn(self._renderer)
        except Exception:  # a render failure must not take down the caller
            log.exception("mesh GPU render failed, falling back to CPU rasterizer")
            return None


_GL = _GLThread()


def gl_available() -> bool:
    """Whether a headless GL context could be created (probes once, then caches)."""
    return _GL.submit(lambda r: True) is True


def render_mesh_rgba_gl(
    verts_world: np.ndarray,
    faces: np.ndarray,
    face_rgb: np.ndarray,
    valid: np.ndarray,
    camera: "Camera",
    height: int,
    width: int,
    *,
    alpha: float = 0.55,
) -> np.ndarray | None:
    """GPU twin of :func:`deeperfly.visualization.mesh.render_mesh_rgba`.

    Returns the same ``(H, W, 4)`` uint8 RGBA overlay, or ``None`` if no GL context
    is available or the camera has lens distortion (the GPU path is the linear
    pinhole, so a distorted camera falls back to the software rasterizer to stay
    exact). Posing a frame once and rendering several cameras reuses the uploaded
    geometry: pass the *same* ``verts_world`` array for each view of a frame.
    """
    height, width = int(height), int(width)
    dist = np.asarray(getattr(camera, "dist", []), dtype=float).reshape(-1)
    if dist.size and np.any(np.abs(dist) > 1e-9):
        return None  # keep distortion: let the software rasterizer handle this camera

    img = _GL.submit(
        lambda r: r.render(
            np.asarray(verts_world, dtype=float),
            np.asarray(faces),
            np.asarray(face_rgb),
            np.asarray(valid),
            camera.intr,
            camera.rmat,
            camera.tvec,
            width,
            height,
        )
    )
    if img is None:
        return None
    cover = img[..., 3] > 0
    img[..., 3] = cover.astype(np.uint8) * int(round(np.clip(alpha, 0, 1) * 255))
    return img
