"""The FastAPI app: serves frames + 2D overlays and applies edits over a socket.

:func:`create_app` wraps a :class:`~deeperfly.gui.session.Session` (the Qt-free
editor model) in HTTP + WebSocket handlers. The browser front-end (``web/``)
fetches metadata and per-frame overlays as JSON, pulls each camera's frame as a
JPEG, and streams edits over ``/ws`` -- every edit maps one-to-one onto an
:class:`~deeperfly.gui.state.EditorState` method and replies with the refreshed
per-view points so the canvases repaint (the same flow the old Qt window drove
with signals). Corrections live only in memory until ``POST /api/save`` writes
the ``corrections.h5`` sidecar.

All state mutations are serialized by a single :class:`asyncio.Lock`: one
operator on one result is the expected case, and the edit ops are fast,
in-process NumPy/JAX, so holding the lock briefly is harmless.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..visualization._palette import point_colors_rgb
from .corrections import save_corrections
from .session import Session

__all__ = ["create_app"]

log = logging.getLogger("deeperfly")

_WEB_DIR = Path(__file__).parent / "web"


def create_app(
    session: Session,
    *,
    on_shutdown: Callable[[], None] | None = None,
    exit_on_disconnect: bool = False,
    disconnect_grace: float = 0.5,
) -> FastAPI:
    """Build the FastAPI app serving and editing ``session``.

    ``on_shutdown``, when given, is invoked to stop the running server -- by
    ``POST /api/shutdown`` (the GUI's Close button) and, when ``exit_on_disconnect``
    is set, ``disconnect_grace`` seconds after the last browser drops its ``/ws``
    socket (closing the tab). The grace is short so a real close stops the server
    almost instantly, yet still lets a page refresh -- which also drops the socket --
    reconnect (fast, on localhost) and cancel the pending shutdown. The browser
    holds exactly one socket open for its whole lifetime, so the live socket count
    tracks open tabs. :func:`deeperfly.gui.serve` passes a callback that flips
    uvicorn's ``should_exit``; tests pass a plain stub.
    """
    app = FastAPI(title="deeperfly gui")
    lock = asyncio.Lock()
    # Open `/ws` sockets (one per browser tab) and the timer that, once the last
    # one closes, stops the server after the grace period (cancelled on reconnect).
    clients = 0
    pending_exit: asyncio.TimerHandle | None = None

    # The web assets are edited in place (no build step), so without an explicit
    # policy a browser's heuristic cache can serve a stale app.js/styles.css
    # against freshly changed HTML -- a half-broken editor. Force revalidation on
    # every load of the page and its assets; the ETag keeps it cheap (a 304 when
    # nothing changed). Frame JPEGs keep their own long max-age (set per-response).
    @app.middleware("http")
    async def _revalidate_assets(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    static_dir = _WEB_DIR / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    else:  # pragma: no cover -- the assets ship in the package, so this is unexpected
        log.warning(
            "web assets missing at %s (expected alongside server.py)", static_dir
        )

    @app.get("/")
    def index() -> Response:
        page = _WEB_DIR / "index.html"
        if not page.is_file():  # pragma: no cover
            raise HTTPException(500, "web/index.html is missing (build the GUI)")
        return FileResponse(page)

    @app.get("/api/meta")
    def meta() -> dict:
        return _meta_payload(session)

    @app.get("/api/frame/{camera}/{t}")
    def frame(camera: str, t: int) -> Response:
        img = session.source.frame(camera, t)
        if img is None:
            raise HTTPException(404, f"no frame for {camera!r} at {t}")
        ok, buf = cv2.imencode(".jpg", _to_bgr(img))
        if not ok:  # pragma: no cover -- encoder failure is not expected
            raise HTTPException(500, "frame encoding failed")
        return Response(
            content=buf.tobytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=3600"},
        )

    mesh_cache: dict[tuple[str, int], bytes] = {}

    @app.get("/api/mesh/{camera}/{t}")
    def mesh(camera: str, t: int) -> Response:
        """The posed NeuroMechFly mesh for ``camera`` at frame ``t`` as an RGBA PNG.

        404 when the result carries no fitted model (IK off). Rendered on demand and
        memoized per ``(camera, frame)`` so scrubbing back is instant; the overlay is
        heavy enough that re-rendering every scrub would lag.
        """
        if not session.state.has_nmf:
            raise HTTPException(404, "no inverse-kinematics model to overlay")
        key = (camera, _clamp_frame(session, t))
        if key not in mesh_cache:
            png = _render_mesh_png(session, camera, key[1])
            if png is None:
                raise HTTPException(404, f"no mesh overlay for {camera!r} at {t}")
            mesh_cache[key] = png
        return Response(
            content=mesh_cache[key],
            media_type="image/png",
            headers={"Cache-Control": "max-age=3600"},
        )

    @app.get("/api/nmf/asset")
    def nmf_asset() -> Response:
        """The static NMF mesh topology + per-vertex colors (binary), for the client."""
        data = _nmf_asset_bytes()
        if data is None or not session.state.has_nmf:
            raise HTTPException(404, "no inverse-kinematics model to overlay")
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"Cache-Control": "max-age=3600"},
        )

    @app.get("/api/nmf/verts/{t}")
    async def nmf_verts(t: int) -> Response:
        """The posed NMF vertices, normals + valid-face mask for ``t`` (re-fit from edits)."""
        if not session.state.has_nmf:
            raise HTTPException(404, "no inverse-kinematics model to overlay")
        async with lock:
            data = _nmf_verts_bytes(session, _clamp_frame(session, t))
        if data is None:
            raise HTTPException(404, f"no mesh overlay at frame {t}")
        return Response(content=data, media_type="application/octet-stream")

    @app.get("/api/points/{t}")
    def points(t: int, mode: str = "view") -> dict:
        return _points_payload(session, _clamp_frame(session, t), mode)

    @app.get("/api/scene/{t}")
    def scene(t: int) -> dict:
        return _scene_payload(session, _clamp_frame(session, t))

    @app.get("/api/corrected")
    def corrected() -> dict:
        """The frames carrying manual corrections (sorted, with per-frame counts).

        Drives the editor's corrected-frames list; the front-end refreshes it after
        edits settle, so it tracks every drag, obscure, and reset live.
        """
        return {"frames": session.state.corrected_frames()}

    @app.post("/api/save")
    async def save() -> dict:
        async with lock:
            save_corrections(
                session.corrections_path,
                session.state.corrections,
                source=session.results_path,
            )
        return {"dirty": session.state.dirty}

    @app.post("/api/shutdown")
    async def shutdown() -> dict:
        """Stop the server (the GUI's Close button).

        The browser saves or discards any unsaved corrections before calling this,
        so the handler touches no state -- it just signals the run loop to exit.
        Returns first; uvicorn finishes this reply, then shuts down on its next
        tick. A no-op (still ``200``) when no shutdown hook was wired in.
        """
        log.info("gui requested shutdown")
        if on_shutdown is not None:
            on_shutdown()
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        nonlocal clients, pending_exit
        await websocket.accept()
        clients += 1
        if pending_exit is not None:
            # A reconnect (typically a page refresh) cancels a pending shutdown.
            pending_exit.cancel()
            pending_exit = None
            log.info("browser reconnected; shutdown cancelled")
        try:
            while True:
                msg = await websocket.receive_json()
                try:
                    async with lock:
                        payload = _handle_edit(session, msg)
                except (KeyError, ValueError, TypeError) as exc:
                    # A malformed edit must not tear down the editing session.
                    log.warning("ignoring bad edit message %r: %s", msg, exc)
                    continue
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        finally:
            clients -= 1
            # The last tab closed: stop the server, but give a refresh's reconnect
            # the grace period to cancel it first.
            if exit_on_disconnect and clients == 0 and on_shutdown is not None:
                log.info("browser disconnected; stopping in %ss", disconnect_grace)
                pending_exit = asyncio.get_running_loop().call_later(
                    disconnect_grace, on_shutdown
                )

    return app


# -- mesh overlay -------------------------------------------------------------


def _render_mesh_png(session: Session, camera: str, t: int) -> bytes | None:
    """Render the posed NMF mesh for ``camera`` at frame ``t`` to RGBA PNG bytes.

    Sized to the camera's footage frame (so it overlays the served frame exactly).
    Returns ``None`` if the model or the packaged mesh asset is unavailable.
    """
    s = session.state
    if s.result.nmf_pts3d is None or camera not in s.result.cameras.names:
        return None
    try:
        from ..inverse_kinematics.mesh import load_nmf_mesh
        from ..visualization.mesh import render_mesh_rgba_auto
    except Exception:  # pragma: no cover -- a missing asset disables the overlay
        return None
    cam = s.result.cameras[camera]
    h, w = session.image_sizes.get(camera) or _intr_size(cam)
    mesh = load_nmf_mesh()
    angles = None if s.result.nmf_angles is None else s.result.nmf_angles[t]
    verts, valid = mesh.pose(
        s.result.nmf_pts3d[t],
        angles,
        s.result.nmf_angle_names,
        head_scale=s.result.nmf_head_scale,
        abdomen_scale=s.result.nmf_abdomen_scale,
        body_scale=s.result.nmf_body_scale,
    )
    valid = np.asarray(valid) & ~mesh.hidden_face_mask(session.nmf_hide_parts)
    rgba = render_mesh_rgba_auto(
        verts, mesh.faces, mesh.face_rgb, valid, cam, int(h), int(w)
    )
    # cv2 writes BGRA; reorder RGBA -> BGRA so the PNG colors are correct.
    ok, buf = cv2.imencode(".png", rgba[..., [2, 1, 0, 3]])
    return buf.tobytes() if ok else None


def _intr_size(cam) -> tuple[int, int]:
    """``(height, width)`` inferred from a camera's principal point (no footage)."""
    intr = np.asarray(cam.intr)
    return int(round(2 * intr[3] + 1)), int(round(2 * intr[2] + 1))


# -- client-rendered mesh (WebGL) ---------------------------------------------
#
# The browser renders the posed NMF mesh on the GPU, so the server only ships the
# geometry: the topology + per-vertex colors once (`/api/nmf/asset`) and the posed
# vertices per frame (`/api/nmf/verts/{t}`, re-fit live from the corrected pose).
# Vertices/faces/colors are little-endian binary so the front-end can drop them
# straight into typed arrays (no megabytes of JSON to parse on every scrub).


@functools.lru_cache(maxsize=2)
def _nmf_asset_bytes() -> bytes | None:
    """The static mesh topology + per-vertex colors, packed once for the client.

    Layout (little-endian): ``uint32 n_verts``, ``uint32 n_faces``,
    ``uint32[n_faces * 3]`` triangle indices, ``uint8[n_verts * 3]`` vertex RGB.
    """
    try:
        from ..inverse_kinematics.mesh import load_nmf_mesh
    except Exception:  # pragma: no cover -- a missing asset disables the overlay
        return None
    mesh = load_nmf_mesh()
    faces = np.asarray(mesh.faces, dtype="<u4")
    n_verts = int(mesh.vertices.shape[0])
    # Per-vertex color from the per-face palette: each vertex belongs to one baked
    # mesh part, so all its faces share a color and the assignment is unambiguous.
    vrgb = np.zeros((n_verts, 3), dtype=np.uint8)
    face_rgb = np.asarray(mesh.face_rgb, dtype=np.uint8)
    for k in range(3):
        vrgb[faces[:, k]] = face_rgb
    header = np.array([n_verts, faces.shape[0]], dtype="<u4")
    return header.tobytes() + faces.tobytes() + vrgb.tobytes()


def _nmf_verts_bytes(session: Session, t: int) -> bytes | None:
    """The posed vertices + smooth normals + valid-face mask for ``t`` (re-fit from edits).

    Layout (little-endian): ``float32[n_verts * 3]`` world vertices (NaN -> 0), then
    ``float32[n_verts * 3]`` smooth per-vertex normals, then ``uint8[n_faces]`` --
    ``1`` where all three of a face's vertices were posed. The normals let the client
    smooth-shade the overlay (no faceting), and are computed here once per frame (the
    head/abdomen size is the IK data estimate, not an operator knob).
    """
    posed = session.state.nmf_posed_verts(t)
    if posed is None:
        return None
    from ..inverse_kinematics.mesh import load_nmf_mesh
    from ..visualization.mesh import vertex_normals

    verts, valid = posed
    mesh = load_nmf_mesh()
    faces = mesh.faces
    # Hide the configured body parts (default: wings) by dropping their faces.
    valid = np.asarray(valid) & ~mesh.hidden_face_mask(session.nmf_hide_parts)
    normals = vertex_normals(np.asarray(verts, dtype=float), faces, valid)
    verts = np.nan_to_num(np.asarray(verts, dtype="<f4"), nan=0.0)
    return (
        verts.tobytes()
        + np.asarray(normals, dtype="<f4").tobytes()
        + np.asarray(valid, dtype=np.uint8).tobytes()
    )


def _cameras_proj(session: Session) -> list[dict]:
    """Each camera's pinhole projection for the client's WebGL overlay.

    ``intr`` is ``[fx, fy, cx, cy]``, ``rmat`` the 3x3 world->camera rotation (row
    major), ``tvec`` its translation, and ``size`` the footage ``[width, height]``
    the intrinsics describe -- enough to build the exact projection
    :meth:`CameraGroup.project` uses (validated to sub-pixel agreement).
    """
    out = []
    for name, cam in zip(session.state.camera_names, session.state.result.cameras):
        h, w = session.image_sizes.get(name) or _intr_size(cam)
        out.append(
            {
                "name": name,
                "intr": [float(v) for v in np.asarray(cam.intr)],
                "rmat": [float(v) for v in np.asarray(cam.rmat).reshape(-1)],
                "tvec": [float(v) for v in np.asarray(cam.tvec)],
                "size": [int(w), int(h)],
            }
        )
    return out


# -- payload builders ---------------------------------------------------------


def _meta_payload(session: Session) -> dict:
    """The one-time metadata the front-end needs to lay out and draw the editor."""
    s = session.state
    skel = s.result.skeleton
    colors = (np.asarray(point_colors_rgb(skel)) * 255).round().astype(int)
    return {
        "results_path": session.results_path,
        "n_views": s.n_views,
        "n_frames": session.n_frames,
        "n_points": s.n_points,
        "has_3d": s.has_3d,
        "has_nmf": s.has_nmf,
        "camera_names": list(s.camera_names),
        "image_sizes": {
            name: [int(h), int(w)] for name, (h, w) in session.image_sizes.items()
        },
        "point_names": list(skel.point_names),
        "bones": np.asarray(skel.bones, dtype=int).reshape(-1, 2).tolist(),
        "point_colors": colors.tolist(),
        "cameras_3d": _cameras_3d(session),
        "cameras_proj": _cameras_proj(session),
        "dirty": bool(s.dirty),
    }


def _cameras_3d(session: Session) -> list[dict]:
    """Each camera's world-frame pose for the on-demand 3D rig plot.

    ``position`` is the camera centre; ``right``/``up``/``forward`` are the unit
    world-frame axes of the camera (the rows of the rotation matrix are +x
    image-right, +y image-down, +z optical, so ``up`` is the negated middle row).
    """
    cams = []
    for name, cam in zip(session.state.camera_names, session.state.result.cameras):
        rmat = np.asarray(cam.rmat)
        cams.append(
            {
                "name": name,
                "position": [float(v) for v in np.asarray(cam.position)],
                "right": [float(v) for v in rmat[0]],
                "up": [float(-v) for v in rmat[1]],
                "forward": [float(v) for v in rmat[2]],
            }
        )
    return cams


def _points_payload(
    session: Session, t: int, mode: str, *, include_nmf: bool = True
) -> dict:
    """The per-view 2D overlay (with the fixed/invisible masks) for frame ``t`` in ``mode``.

    ``proj`` is the current 3D estimate reprojected into every view (with no fixed
    overrides) -- the display-only "latent skeleton" the front-end can ghost over
    every view; it is ``null`` when the result carries no 3D points.

    ``include_nmf`` (default ``True``) controls whether the fitted-model overlay is
    computed. The NMF reprojection needs a per-frame inverse-kinematics re-fit, so a
    live 3D drag -- which streams one edit per animation frame -- passes ``False`` to
    skip it (the ``nmf`` key is then *omitted*, and the front-end keeps the overlay it
    has until the drag settles). It is recomputed on the pin/settle reply and on plain
    fetches.
    """
    s = session.state
    if mode == "edit_3d" and s.has_3d:
        pts = s.display_pts2d_refine(t)
    else:
        pts = s.display_pts2d(t)
    fixed = s.corrections.pts2d_fixed[:, t]  # (V, P)
    invisible = s.corrections.pts2d_invisible[:, t]  # (V, P)
    proj = s.display_pts3d_projected(t) if s.has_3d else None
    payload = {
        "frame": t,
        "mode": mode,
        "points": _points_to_json(np.asarray(pts)),
        "fixed": fixed.tolist(),
        "invisible": invisible.tolist(),
        "proj": None if proj is None else _points_to_json(np.asarray(proj)),
        "dirty": bool(s.dirty),
    }
    if include_nmf:
        nmf = s.display_nmf_projected(t) if s.has_nmf else None
        payload["nmf"] = None if nmf is None else _points_to_json(np.asarray(nmf))
    return payload


def _scene_payload(session: Session, t: int) -> dict:
    """The frame's 3D pose for the scene view: the triangulated keypoints and the
    fitted NMF model joints (both world frame, skeleton order), each ``null`` when
    unavailable (2D-only, or no inverse-kinematics model)."""
    s = session.state
    pts3d = s.display_pts3d(t)
    nmf = s.nmf_fit(t) if s.has_nmf else None
    nmf3d = None if nmf is None else np.asarray(nmf[0])
    return {
        "frame": t,
        "points3d": None if pts3d is None else _points3d_to_json(np.asarray(pts3d)),
        "nmf3d": None if nmf3d is None else _points3d_to_json(nmf3d),
    }


def _points_to_json(pts: np.ndarray) -> list:
    """``(V, P, 2)`` points to nested lists, with ``null`` for any NaN point."""
    finite = np.isfinite(pts).all(axis=-1)
    return [
        [
            [float(pts[v, p, 0]), float(pts[v, p, 1])] if finite[v, p] else None
            for p in range(pts.shape[1])
        ]
        for v in range(pts.shape[0])
    ]


def _points3d_to_json(pts: np.ndarray) -> list:
    """``(P, 3)`` world points to nested lists, with ``null`` for any NaN point."""
    finite = np.isfinite(pts).all(axis=-1)
    return [
        [float(pts[p, 0]), float(pts[p, 1]), float(pts[p, 2])] if finite[p] else None
        for p in range(pts.shape[0])
    ]


# -- edit dispatch ------------------------------------------------------------


def _handle_edit(session: Session, msg: dict) -> dict:
    """Apply one edit message to the state and return the refreshed points payload.

    Each ``type`` maps to an :class:`~deeperfly.gui.state.EditorState` op; the
    reply is the standard points payload for the message's ``frame`` and ``mode``
    so the client repaints every view (e.g. a live 3D drag moving all views).
    """
    s = session.state
    t = _clamp_frame(session, int(msg.get("frame", 0)))
    mode = str(msg.get("mode", "view"))
    typ = msg.get("type")
    # A live 3D drag (edit_3d with fix=False) streams one edit per animation frame.
    # The NMF overlay reprojection needs a per-frame IK re-fit, so recompute it only
    # on the pin/settle reply -- not ~60x/s mid-drag (the dominant source of drag lag).
    live_drag = typ == "edit_3d" and not bool(msg.get("fix", False))
    if typ == "edit_2d":
        s.apply_2d_edit(int(msg["view"]), int(msg["point"]), _xy(msg), t)
    elif typ == "edit_3d":
        s.apply_3d_edit(
            int(msg["view"]),
            int(msg["point"]),
            _xy(msg),
            t,
            fix=bool(msg.get("fix", False)),
        )
    elif typ == "toggle_fixed":
        s.toggle_fixed(int(msg["view"]), int(msg["point"]), t)
    elif typ == "toggle_invisible":
        s.toggle_invisible(int(msg["view"]), int(msg["point"]), t)
    elif typ == "reset_point":
        s.reset_point(int(msg["point"]), t)
    elif typ == "reset_point_view":
        s.reset_point_view(int(msg["view"]), int(msg["point"]), t)
    elif typ == "reset_frame":
        s.reset_frame(t)
    else:  # pragma: no cover -- an unknown type is a client bug; ignore it
        log.warning("ignoring unknown edit message type %r", typ)
    payload = _points_payload(session, t, mode, include_nmf=not live_drag)
    # Echo the client's monotonic edit seq (when present) so the front-end can drop a
    # superseded reply -- a mid-drag re-solve that lands after release would otherwise
    # repaint the joint to a stale position (the snap-back).
    payload["seq"] = msg.get("seq")
    return payload


def _xy(msg: dict) -> tuple[float, float]:
    return (float(msg["x"]), float(msg["y"]))


def _clamp_frame(session: Session, t: int) -> int:
    """Keep ``t`` inside ``[0, n_frames)`` (defensive against stray indices)."""
    return max(0, min(int(t), session.n_frames - 1))


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """An ``(H, W, 3)`` RGB (or ``(H, W)`` gray) frame as the BGR cv2 expects.

    ``FrameSource`` yields RGB; ``cv2.imencode`` reads BGR, so the channels are
    reversed (a contiguous copy) before encoding to keep colors correct in the
    browser. Grayscale frames pass straight through.
    """
    if img.ndim == 2:
        return img
    return np.ascontiguousarray(img[..., ::-1])
