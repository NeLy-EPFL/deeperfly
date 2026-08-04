"""The FastAPI app: serves frames + 2D overlays and applies edits over a socket.

:func:`create_app` wraps a :class:`~deeperfly.gui.session.Session` (the Qt-free
editor model) in HTTP + WebSocket handlers. The browser front-end (``web/``)
fetches metadata and per-frame overlays as JSON, pulls each camera's frame as a
JPEG, and streams edits over ``/ws`` -- every edit maps one-to-one onto an
:class:`~deeperfly.gui.state.EditorState` method and replies with the refreshed
per-view points so the canvases repaint (the same flow the old Qt window drove
with signals). Corrections live only in memory until ``POST /api/save`` writes
the ``corrections.h5`` sidecar.

All state mutations are serialized by a single :class:`asyncio.Lock`, and only
one connected browser -- the "writer" -- may edit at a time. The session is one
shared :class:`~deeperfly.gui.state.EditorState`, so two tabs editing at once
would silently overwrite each other's corrections; instead the first ``/ws``
socket to open holds the writer slot and every later socket is read-only. A
read-only tab can take over (a ``{"type": "claim"}`` message) or is promoted
automatically when the writer disconnects. The edit ops are fast, in-process
NumPy/JAX, so holding the lock briefly is harmless.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

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
from fastapi.staticfiles import StaticFiles

from ..acquisition import read_suggestions, suggestions_staleness
from ..visualization._palette import point_colors_rgb
from .labels import save_labels
from .session import Session

if TYPE_CHECKING:
    from ..skeleton import Skeleton

__all__ = ["create_app"]

log = logging.getLogger("deeperfly")

_WEB_DIR = Path(__file__).parent / "web"


def _asset_version() -> str:
    """A short content hash of the entry assets, stamped into the URLs the page loads.

    ``index.html`` references ``app.js`` / ``styles.css`` as ``...?v=<hash>``. The page
    itself is served ``no-cache`` (revalidated every load), so a fresh load always
    carries the current hash and the browser fetches the *matching* JS/CSS. Without
    this, a browser that serves a heuristically-cached ``app.js`` against freshly
    changed HTML gets a half-broken editor -- the stale JS wires to DOM ids the new
    HTML no longer has. Computed per request (cheap) so in-place edits take effect on
    the next reload with no server restart."""
    h = hashlib.sha1()
    for name in ("app.js", "styles.css"):
        p = _WEB_DIR / "static" / name
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _session_version(session: Session) -> str:
    """A short token identifying *which pictures* this session serves.

    Frame JPEGs and mesh overlays are addressed only by camera and frame index, so
    ``/api/frame/f/1506`` names a different picture in every recording while looking
    identical to an HTTP cache. Every ``deeperfly gui`` runs on the same default
    ``127.0.0.1:8000``, and the suggestion queue picks near-identical frame indices
    across recordings of the same length -- so opening one recording after another
    within the cached lifetime made the browser paint the *previous* recording's fly.
    Stamping this token into the URLs puts the recording into the cache key, which is
    what makes the long ``max-age`` safe.

    Keyed on the resolved footage (identity and bytes-on-disk), not on ``results.h5``:
    the footage is what the served pixels actually come from, and it does not change
    as the operator labels -- whereas an absence declaration rewrites ``results.h5``,
    which would needlessly retire every cached frame. Re-pointing a result at
    different videos does change the token, so those frames are refetched.
    """
    parts = [str(Path(session.results_path).resolve())]
    for name, files in sorted(session.source.footage_files.items()):
        for path in files:
            try:
                st = path.stat()
                parts.append(f"{name}:{path.resolve()}:{st.st_size}:{st.st_mtime_ns}")
            except OSError:  # pragma: no cover -- resolved footage exists
                parts.append(f"{name}:{path}")
    return hashlib.sha1("\0".join(parts).encode()).hexdigest()[:12]


def create_app(
    session: Session,
    *,
    on_shutdown: Callable[[], None] | None = None,
    exit_on_disconnect: bool = False,
    disconnect_grace: float = 0.5,
    jobs=None,
) -> FastAPI:
    """Build the FastAPI app serving and editing ``session``.

    ``jobs``, when given, is a :class:`~deeperfly.jobs.JobQueue` the browser may submit
    work to (``/api/jobs``). ``None`` -- the default, and what a bare ``results.h5``
    session gets -- leaves the endpoints reporting ``enabled: false`` rather than absent,
    so the front-end can say *why* the buttons are missing instead of just not showing
    them.

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
    # Computed once: it stats the footage (on a network share, for this lab), and a
    # single value keeps the token the page stamps into its URLs identical to the one
    # the frame handler validates against.
    cache_v = _session_version(session)
    # Open `/ws` sockets (one per browser tab), the single "writer" allowed to edit
    # the shared session, and the timer that -- once the last socket closes -- stops
    # the server after the grace period (cancelled on reconnect).
    sockets: set[WebSocket] = set()
    writer: WebSocket | None = None
    pending_exit: asyncio.TimerHandle | None = None

    def _role_msg(ws: WebSocket) -> dict:
        """The role handshake for a browser: may it edit (writer) or is it read-only?"""
        return {
            "type": "role",
            "role": "writer" if ws is writer else "reader",
            "clients": len(sockets),
        }

    # The web assets are edited in place (no build step), so without an explicit
    # policy a browser's heuristic cache can serve a stale app.js/styles.css
    # against freshly changed HTML -- a half-broken editor. Force revalidation on
    # every load of the page and its assets; the ETag keeps it cheap (a 304 when
    # nothing changed). Frame JPEGs and mesh overlays keep their own long max-age,
    # but only when their URL carries this session's `_session_version` stamp.
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
        html = page.read_text(encoding="utf-8").replace("__ASSET_V__", _asset_version())
        return Response(content=html, media_type="text/html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        # Served from the root path (not only /static) because Safari probes
        # /favicon.ico directly, and Safari is the browser that needs the .ico:
        # it doesn't render the SVG favicon, so without this it falls back to a
        # letter placeholder in the tab.
        ico = _WEB_DIR / "static" / "favicon.ico"
        if not ico.is_file():  # pragma: no cover -- the icon ships in the package
            raise HTTPException(404, "favicon.ico is missing")
        return Response(
            content=ico.read_bytes(),
            media_type="image/x-icon",
            headers={"Cache-Control": "max-age=3600"},
        )

    @app.get("/apple-touch-icon.png", include_in_schema=False)
    def apple_touch_icon() -> Response:
        # Safari probes /apple-touch-icon.png at the root and uses it for its
        # Start Page / dock tiles, where it draws the icon on a rounded chip. The
        # asset is a full-bleed teal tile so it fills that chip cleanly instead of
        # leaving a light box around a circle.
        png = _WEB_DIR / "static" / "apple-touch-icon.png"
        if not png.is_file():  # pragma: no cover -- the icon ships in the package
            raise HTTPException(404, "apple-touch-icon.png is missing")
        return Response(
            content=png.read_bytes(),
            media_type="image/png",
            headers={"Cache-Control": "max-age=3600"},
        )

    @app.get("/api/meta")
    def meta() -> dict:
        return _meta_payload(session, cache_v)

    @app.get("/api/schema")
    def schema(section: str | None = None) -> dict:
        """The settable config keys, their defaults and their documentation.

        Derived from the ``*Params`` dataclasses (:mod:`deeperfly.config_schema`), never
        from a parallel description -- so a new field appears in the GUI's forms on the
        next reload with no second place to update, and its help text is the prose already
        written for it.

        Sections that have no schema (the detection plan, cameras, the skeleton, video
        specs) are open-ended and are reported as such rather than shown as empty forms.
        """
        from ..config_schema import describe, sections, stage_flags_spec

        if section is not None:
            try:
                return describe(section).as_dict()
            except KeyError as exc:
                raise HTTPException(404, str(exc).strip("'")) from None
        return {
            "sections": [stage_flags_spec().as_dict()]
            + [describe(name).as_dict() for name in sections()],
            # Named so a form builder can say "this one needs the file" instead of
            # rendering nothing and looking broken.
            "undescribable": [
                "cameras",
                "skeleton",
                "sources",
                "pose2d.models",
                "pose2d.pathways",
                "pose2d.output_points",
                "visualization.videos",
            ],
        }

    def _image_cache_control(v: str | None) -> str:
        """Long-lived only for URLs stamped with *this* session's token.

        A stamped URL can never collide with another recording's, so it is safe to
        keep (and ``immutable`` spares the revalidation on scrub-back). An unstamped
        or stale-stamped request is not addressed to a unique picture, so it must not
        enter the cache at all -- see :func:`_session_version`."""
        return "max-age=3600, immutable" if v == cache_v else "no-store"

    @app.get("/api/frame/{camera}/{t}")
    def frame(camera: str, t: int, v: str | None = None) -> Response:
        img = session.source.frame(camera, t)
        if img is None:
            raise HTTPException(404, f"no frame for {camera!r} at {t}")
        ok, buf = cv2.imencode(".jpg", _to_bgr(img))
        if not ok:  # pragma: no cover -- encoder failure is not expected
            raise HTTPException(500, "frame encoding failed")
        return Response(
            content=buf.tobytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": _image_cache_control(v)},
        )

    mesh_cache: dict[tuple[str, int], bytes] = {}

    @app.get("/api/mesh/{camera}/{t}")
    def mesh(camera: str, t: int, v: str | None = None) -> Response:
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
            headers={"Cache-Control": _image_cache_control(v)},
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
    def points(t: int, mode: str = "view", verbose: bool = False) -> dict:
        return _points_payload(session, _clamp_frame(session, t), mode, verbose=verbose)

    @app.get("/api/scene/{t}")
    def scene(t: int) -> dict:
        return _scene_payload(session, _clamp_frame(session, t))

    @app.get("/api/corrected")
    def corrected() -> dict:
        """The frames the operator has touched (sorted), each with its reviewed flag.

        Drives the editor's corrected-frames list; the front-end refreshes it after
        edits settle, so it tracks every drag, obscure, reset, and reviewed tick live.
        """
        return {"frames": session.state.corrected_frames()}

    @app.get("/api/suggestions")
    def suggestions() -> dict:
        """The ranked "label these next" queue, joined with the live editing state.

        Serves the ``labels_suggest.json`` sidecar (see :func:`_suggestions_payload`)
        -- never recomputes it: the ranking triangulates the whole recording, which
        costs seconds, so it is a CLI artefact and this route is a reader. Always
        ``200``: no sidecar is a normal state (``present: false``), not an error.
        """
        return _suggestions_payload(session)

    # -- config ---------------------------------------------------------------
    #
    # The values come from the project's composed config; a write goes to its PROFILE, which
    # holds only what differs from the packaged defaults. So the editor and the CLI change
    # the same file the same way, and "reset to default" genuinely removes the key rather
    # than restating the default as though someone had chosen it.

    def _project():
        if session.project_root is None:
            raise HTTPException(
                409,
                "this session was opened on a bare results.h5; open a project to change "
                "its settings from the editor",
            )
        from ..project import Project

        return Project.load(session.project_root)

    @app.get("/api/config")
    def get_config() -> dict:
        """Every describable section's current values, with which ones were actually set."""
        from ..config_schema import effective, sections, stage_flags_spec

        if session.project_root is None:
            return {"enabled": False, "reason": "no project", "sections": {}}
        project = _project()
        import tomllib

        from ..config import Config

        try:
            config = Config.from_dict(tomllib.loads(project.compose_config()))
        except Exception as exc:
            raise HTTPException(500, f"the project's config does not compose: {exc}")
        overrides = project.profile_values()
        out: dict = {}
        for name in sections():
            try:
                values = effective(config, name)
            except (
                Exception
            ) as exc:  # a malformed section must not blank the whole panel
                out[name] = {"error": str(exc)}
                continue
            out[name] = {
                field: {
                    "value": _jsonable(value),
                    "is_default": is_default,
                    "overridden": field in (overrides.get(name) or {}),
                }
                for field, (value, is_default) in values.items()
            }
        flags = config.stage_flags()
        out["pipeline"] = {
            f.name: {
                "value": flags.get(f.name.removeprefix("do_"), f.default),
                "is_default": f.name not in (overrides.get("pipeline") or {}),
                "overridden": f.name in (overrides.get("pipeline") or {}),
            }
            for f in stage_flags_spec().fields
        }
        return {
            "enabled": True,
            "profile": project.profile_path().name,
            "sections": out,
        }

    @app.post("/api/config")
    def set_config(payload: dict) -> dict:
        """Set one ``{"section", "key", "value"}`` in the project's profile.

        ``value: null`` clears the override. Validated through ``Config``'s own strict
        loader, so the editor cannot store a key a run would reject.
        """
        project = _project()
        section, key = str(payload.get("section", "")), str(payload.get("key", ""))
        try:
            path = project.set_profile_key(section, key, payload.get("value"))
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from None
        log.info("set %s.%s in %s", section, key, path.name)
        return {"ok": True, "profile": str(path)}

    # -- jobs ----------------------------------------------------------------
    #
    # Polled by the panel rather than pushed over `/ws`. The socket carries the *editing*
    # stream and is single-writer by design; a read-only tab must still see the queue, and
    # a broadcast would either bypass that lock or duplicate it. A 2 s poll of a
    # handful-of-rows JSON payload is cheaper than the complexity.

    @app.get("/api/jobs")
    def list_jobs(tail: int = 5) -> dict:
        """The queue, newest first. ``enabled: false`` when this session has no queue."""
        if jobs is None:
            return {
                "enabled": False,
                "reason": "this session was opened on a bare results.h5; open a project "
                "to run jobs from the editor",
                "jobs": [],
            }
        return {
            "enabled": True,
            "busy": jobs.busy,
            "jobs": [j.as_dict(tail=tail) for j in jobs.list()],
        }

    @app.post("/api/jobs")
    async def submit_job(payload: dict) -> dict:
        """Queue a job. ``{"kind": ..., "argv": [...], "label": ..., "recording": ...}``.

        Only allow-listed kinds are accepted (:data:`~deeperfly.jobs.JOB_KINDS`) -- the
        editor is reachable over HTTP, and a queue that ran arbitrary argv would be a
        remote shell.
        """
        if jobs is None:
            raise HTTPException(409, "this session has no job queue (open a project)")
        try:
            job = jobs.submit(
                str(payload.get("kind", "")),
                [str(a) for a in (payload.get("argv") or [])],
                label=str(payload.get("label", "")),
                recording=payload.get("recording"),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return job.as_dict()

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str, tail: int = 200) -> dict:
        if jobs is None:
            raise HTTPException(409, "this session has no job queue")
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id}")
        return job.as_dict(tail=tail)

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: str) -> dict:
        if jobs is None:
            raise HTTPException(409, "this session has no job queue")
        if jobs.get(job_id) is None:
            raise HTTPException(404, f"no job {job_id}")
        return {"cancelled": jobs.cancel(job_id)}

    @app.post("/api/save")
    async def save() -> dict:
        async with lock:
            save_labels(
                session.labels_path,
                session.state.labels,
                identity=session.identity,
                subject_id=session.state.labels.subject_id,
                # Passed back explicitly: save_labels rewrites the whole file, so omitting
                # them would drop the landmarks group the solve reads.
                landmarks=session.state.landmarks,
            )
            # Mirror the absence declaration into results.h5's `animal/` group. That is the
            # seam the pipeline and every results.h5-only consumer read, so a fact authored
            # here reaches a re-run (and the render path) without anyone parsing labels.h5.
            # An in-place patch, so no stage output is touched.
            try:
                from ..results import StageStore

                StageStore(Path(session.results_path)).write_animal(
                    absent=session.state.labels.absent_all_frames(),
                    subject_id=session.state.labels.subject_id,
                )
            except Exception:  # a read-only results.h5 must not fail the label save
                log.exception(
                    "could not mirror the absence declaration into results.h5"
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
        nonlocal writer, pending_exit
        await websocket.accept()
        sockets.add(websocket)
        if pending_exit is not None:
            # A reconnect (typically a page refresh) cancels a pending shutdown.
            pending_exit.cancel()
            pending_exit = None
            log.info("browser reconnected; shutdown cancelled")
        if writer is None:
            # The first (sole) editor. It is told nothing -- an unadorned client is
            # editable by default -- so the single-browser flow (and its tests) is
            # unchanged: the first message it receives is still its own edit reply.
            writer = websocket
        else:
            # A second+ browser: read-only until it takes over or the writer leaves.
            await websocket.send_json(_role_msg(websocket))
        try:
            while True:
                msg = await websocket.receive_json()
                if msg.get("type") == "claim":
                    # This browser takes over editing (the read-only "Take over"
                    # action). The slot is single-valued, so the previous writer,
                    # if any, is demoted to read-only.
                    old = writer
                    writer = websocket
                    if old is not None and old is not websocket:
                        await old.send_json(_role_msg(old))
                    await websocket.send_json(_role_msg(websocket))
                    continue
                if websocket is not writer:
                    # A read-only browser must not mutate the shared session: refuse
                    # the edit and re-assert its role (its UI already blocks this, so
                    # this is the belt-and-braces server guard).
                    log.info("ignoring edit from a read-only browser")
                    await websocket.send_json(_role_msg(websocket))
                    continue
                try:
                    async with lock:
                        payload = _handle_edit(session, msg)
                except (KeyError, ValueError, TypeError, IndexError) as exc:
                    # A malformed edit must not tear down the editing session --
                    # including an out-of-range view/point index in a batched
                    # confirm/reset/occlude `targets` list (an IndexError from the
                    # underlying numpy overlay indexing).
                    log.warning("ignoring bad edit message %r: %s", msg, exc)
                    continue
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        finally:
            sockets.discard(websocket)
            if websocket is writer:
                # The editor left; hand the writer slot to another open browser so a
                # surviving viewer can edit. Left free when no socket remains, so a
                # lone tab's refresh reclaims it on reconnect (writer is None again).
                writer = None
                for cand in list(sockets):
                    writer = cand
                    try:
                        await cand.send_json(_role_msg(cand))
                    except Exception:  # pragma: no cover -- the socket is closing
                        writer = None
                        continue
                    break
            # The last tab closed: stop the server, but give a refresh's reconnect
            # the grace period to cancel it first.
            if exit_on_disconnect and not sockets and on_shutdown is not None:
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
    if session.state.result.cameras is None:
        return []  # uncalibrated: nothing to project through
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


def _limb_legend(skel: Skeleton, colors: np.ndarray) -> list[dict]:
    """Per-limb ``{name, color}`` swatches for the client's colour legend.

    Derived straight from the skeleton's limbs and palette (``colors`` is the
    per-point RGB already computed for the overlay), so the legend reflects
    whatever the loaded config defines -- there is no left/right assumption baked
    into the front-end. Each limb's swatch is the colour of its first point.
    """
    limb_id = np.asarray(skel.limb_id)
    out: list[dict] = []
    for lid, name in enumerate(skel.limb_names):
        members = np.where(limb_id == lid)[0]
        rgb = colors[members[0]] if len(members) else np.array([136, 136, 136])
        out.append({"name": name, "color": [int(c) for c in rgb]})
    return out


def _meta_payload(session: Session, cache_v: str | None = None) -> dict:
    """The one-time metadata the front-end needs to lay out and draw the editor.

    ``cache_v`` is the recording token the server will validate frame URLs against;
    it is passed in (rather than recomputed) so the two can never disagree.
    """
    s = session.state
    skel = s.result.skeleton
    colors = (np.asarray(point_colors_rgb(skel)) * 255).round().astype(int)
    return {
        "results_path": session.results_path,
        # Stamped into the frame/mesh URLs so one recording's images can never be
        # served from cache for another -- see `_session_version`.
        "cache_v": cache_v if cache_v is not None else _session_version(session),
        "n_views": s.n_views,
        "n_frames": session.n_frames,
        "n_points": s.n_points,
        "has_3d": s.has_3d,
        "has_nmf": s.has_nmf,
        # False = no rig has been solved for this recording, so every view is an
        # independent 2D canvas: no reprojection, no derived 3D, no cross-view help. The
        # front-end shows this as a banner rather than leaving the missing overlays
        # unexplained.
        "has_cameras": s.has_cameras,
        # Whether this session can run pipeline commands (a project session can; a bare
        # results.h5 cannot). Reported so the UI can explain an absent button.
        "has_jobs": session.project_root is not None,
        "project_root": None
        if session.project_root is None
        else str(session.project_root),
        "recording": session.recording_slug,
        # Calibration landmarks the project declares. Their own namespace, never the
        # skeleton's: a landmark must not reach the detector, the IK plan or the training
        # export, and must not perturb the fingerprinted point_names.
        "landmarks": _landmarks_meta(session),
        "camera_names": list(s.camera_names),
        "image_sizes": {
            name: [int(h), int(w)] for name, (h, w) in session.image_sizes.items()
        },
        "point_names": list(skel.point_names),
        "bones": np.asarray(skel.bones, dtype=int).reshape(-1, 2).tolist(),
        "point_colors": colors.tolist(),
        "limbs": _limb_legend(skel, colors),
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
    if session.state.result.cameras is None:
        return []  # uncalibrated: there is no rig to plot
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
    session: Session,
    t: int,
    mode: str,
    *,
    include_nmf: bool = True,
    verbose: bool = False,
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
    # Once the frame has an annotation skeleton, IT is the primary layer -- GT pixels over
    # the derived position -- and the detections drop to a reference overlay. Before that
    # there is nothing authored, so the old GT-over-detection view is what to show.
    instance = s.display_instance_pts2d(t)
    if instance is not None:
        pts = instance
    elif mode == "edit_3d" and s.has_3d:
        pts = s.display_pts2d_refine(t)
    else:
        pts = s.display_pts2d(t)
    # Wire-compat masks: "fixed" now means "carries a GT pixel", "invisible" means
    # "occluded" -- the front-end still renders them as the finalized/obscured rings
    # until the Phase-C source-aware rendering lands.
    fixed = s.gt_mask(t)  # (V, P)
    invisible = s.occluded_mask(t)  # (V, P)
    # Absence is per-(frame, point), but this frame's row is broadcast to (V, P) so the
    # front-end indexes it exactly like `fixed` / `invisible`. It must ride the lean
    # mid-drag reply too: it gates whether a joint is drawn at all, so omitting it would
    # flash the phantom limb back on for the duration of every drag.
    absent = np.broadcast_to(s.absent_mask(t)[None, :], fixed.shape)
    proj = s.display_pts3d_projected(t) if s.has_3d else None
    landmarks = s.display_landmarks(t)
    payload = {
        "frame": t,
        "mode": mode,
        "landmarks": None
        if landmarks is None
        else _points_to_json(np.asarray(landmarks)),
        "points": _points_to_json(np.asarray(pts)),
        "instance": None if instance is None else _points_to_json(np.asarray(instance)),
        "has_instance": s.has_instance(t),
        "fixed": fixed.tolist(),
        "invisible": invisible.tolist(),
        "absent": np.asarray(absent).tolist(),
        # Which of those are absent in EVERY frame, so the front-end can tell an
        # amputation from a single-frame declaration without fetching the whole mask.
        "absent_recording": s.absent_points(),
        "proj": None if proj is None else _points_to_json(np.asarray(proj)),
        "dirty": bool(s.dirty),
        "can_undo": s.can_undo,
        "can_redo": s.can_redo,
    }
    # The heavier per-point fields (detector confidence; raw predictions and the
    # missing-joint placeholder seeds for the verbose overlay) are static enough within
    # a frame to ride only the settle/plain reply -- not the ~60x/s mid-drag stream --
    # and the raw prediction / placeholder only when the verbose overlay is on.
    if include_nmf:
        nmf = s.display_nmf_projected(t) if s.has_nmf else None
        payload["nmf"] = None if nmf is None else _points_to_json(np.asarray(nmf))
        payload["conf"] = _conf_to_json(s.result.conf, t)
        payload["chirality"] = _chirality_to_json(s, t)
    if verbose:
        # `s.detections`, not `result.pts2d`: the Detected overlay has to be what the network
        # said. `result.pts2d` is the triangulation-cleaned array and, in a directory prepared
        # for contralateral labeling, has reprojected geometry written over 43% of its finite
        # cells (see EditorState.detections). The solve and the primary layer were switched to
        # `detections`; this reference overlay was missed, so the layer labelled "Detected" was
        # drawing something else.
        payload["pred"] = _points_to_json(np.asarray(s.detections[:, t]))
        # Draggable seeds for joints unobserved in a view (no detection / reprojection),
        # so a GT can still be placed where triangulation dropped the point. Depends on
        # the frame's GT / occlusion / 3D state, so -- unlike `pred` -- it refreshes on
        # every settle/discrete edit (which request verbose), not just on navigation.
        payload["placeholder"] = _points_to_json(np.asarray(s.placeholder_pts2d(t)))
    return payload


def _jsonable(value):
    """A config value as JSON. NumPy scalars and tuples appear via the dataclass defaults."""
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _landmarks_meta(session: Session) -> list[dict]:
    """The project's landmark definitions, for the editor's landmark panel.

    Empty when the project declares none, which is the common case for an established rig
    -- so the panel hides itself rather than offering an empty list.
    """
    state = session.state
    if not state.has_landmarks:
        return []
    counts = state.landmark_counts()
    static = list(state.landmarks.static)  # type: ignore[union-attr]
    return [
        {
            "name": name,
            "static": bool(static[i]),
            "observations": int(counts.get(name, 0)),
        }
        for i, name in enumerate(state.landmark_names())
    ]


def _conf_to_json(conf: np.ndarray | None, t: int) -> list | None:
    """``(V, P)`` detector confidence for frame ``t`` as nested lists (null if absent)."""
    if conf is None:
        return None
    c = np.asarray(conf[:, t], dtype=float)  # (V, P)
    return [
        [float(c[v, p]) if np.isfinite(c[v, p]) else None for p in range(c.shape[1])]
        for v in range(c.shape[0])
    ]


def _chirality_to_json(session_state, t: int) -> dict:
    """The frame's left/right swap verdict, for the editor's warning strip.

    Rides the settle/plain reply alongside ``conf`` rather than the ~60x/s mid-drag stream:
    it needs the frame's derived 3D, and a warning that flickers during a drag is worse
    than one that appears when the drag lands.

    ``points`` carries index pairs *and* names, because the front-end draws the ring from
    the indices and the operator reads the names.
    """
    v = session_state.chirality(t)
    return {
        "decided": bool(v.decided),
        "reason": v.reason,
        "swapped": [
            {
                "points": list(c.points),
                "names": [session_state.point_name(p) for p in c.points],
                "margin": round(float(c.margin), 4),
                "relative_margin": round(float(c.relative_margin), 3),
            }
            for c in v.swapped
        ],
        "n_pairs": int(v.n_pairs),
        "separation_frac": round(float(v.separation_frac), 4),
    }


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


# -- frame suggestions --------------------------------------------------------
#
# `deeperfly labels-suggest` ranks which frames a human should correct next -- by the
# MULTI-VIEW DISAGREEMENT of the detector's own 2D, not by its confidence -- and writes
# the ranked queue to a JSON sidecar beside `results.h5`. The GUI only ever READS that
# file: the ranking triangulates every frame of the recording (seconds), which is fine
# for a CLI and hopeless per HTTP request, so there is no "compute" path here.
#
# The queue is therefore a snapshot, and the panel's honesty about that is this module's
# job, not the front-end's. Every payload carries (a) how stale the sidecar is and (b)
# which of its frames the operator has since labeled, taken from the SAME
# `corrected_frames()` the Labels list uses, so the two lists can never disagree.
# `_suggestions_notes` lifts the facts most likely to mislead (an under-delivered count, a
# reseeded result, uncalibrated cameras) out of the CLI log and onto the screen.
#
# `deeperfly.acquisition` owns the sidecar format, so both parsing (`read_suggestions`)
# and the staleness tiers (`suggestions_staleness`) are ITS functions, imported here
# rather than reimplemented -- a second interpretation of the format is exactly how a
# panel ends up quietly disagreeing with the file it is displaying. Nothing here writes:
# `results.h5` and `labels.h5` are not opened at all, and the sidecar is opened read-only.


def _suggestions_payload(session: Session) -> dict:
    """The suggestion queue for ``session``, joined with its live editing state.

    ``present`` is ``False`` when there is no usable sidecar; ``command`` is then the
    exact command that would produce one (with the resolved directory), so the panel
    can tell the operator what to run rather than just look empty. Entries pointing
    outside the playable frame range are dropped -- a row that cannot be navigated to
    is worse than a missing row -- and counted in the notes.
    """
    path = session.suggestions_path
    command = f"deeperfly labels-suggest {Path(session.results_path).parent}"
    data = _read_suggestions(path)
    if data is None:
        return {
            "present": False,
            "path": None if path is None else str(path),
            "command": command,
        }
    # Live editing state wins over anything the sidecar recorded: `labeled` must mean
    # "has GT now", so a frame flips to done the moment the operator labels it.
    live = {
        int(f["frame"]): bool(f["reviewed"]) for f in session.state.corrected_frames()
    }
    entries = [e for e in (data.get("frames") or []) if isinstance(e, dict)]
    if all(isinstance(e.get("rank"), int) for e in entries):
        entries.sort(key=lambda e: e["rank"])
    frames: list[dict] = []
    n_out_of_range = 0
    for entry in entries:
        try:
            t = int(entry["frame"])
        except (KeyError, TypeError, ValueError):
            continue  # a malformed row is dropped, not fatal
        if not 0 <= t < session.n_frames:
            n_out_of_range += 1
            continue
        frames.append(_suggestion_entry(entry, t, live))
    n_done = sum(1 for f in frames if f["labeled"])
    return {
        "present": True,
        "path": None if path is None else str(path),
        "command": command,
        "computed_utc": data.get("created_utc"),
        "params": data.get("params") or {},
        "source": data.get("source") or {},
        "coverage": data.get("coverage") or {},
        "shortfall": data.get("shortfall") or {},
        # Staleness is judged against the rows actually served, not the raw file, so a
        # malformed row can never reach the helper (whose per-row access assumes a dict).
        "stale": _suggestions_stale(session, {**data, "frames": frames}, n_done=n_done),
        "notes": _suggestions_notes(data, n_out_of_range=n_out_of_range),
        "n_done": n_done,
        "frames": frames,
    }


def _suggestion_entry(entry: dict, t: int, live: dict[int, bool]) -> dict:
    """One queue row: the sidecar's ranking fields plus its live labeled/reviewed state.

    The three quantities the panel formats are coerced to a number or ``None`` here
    rather than passed through: the sidecar is a hand-editable JSON file, and a string
    where a float belongs would otherwise reach the front-end's ``toFixed`` and throw
    mid-render, taking the whole list down over one bad field.

    ``reason`` is passed through verbatim -- it is the "why was I sent here" the
    operator reads (the driving joints, the worst view, and a one-line summary), and the
    front-end only renders its ``summary``, so a richer future reason needs no change
    here.
    """
    return {
        "rank": entry.get("rank"),
        "frame": t,
        "t_s": _number(entry.get("t_s")),
        "score": _number(entry.get("score")),
        "percentile": _number(entry.get("percentile")),
        "kind": entry.get("kind") or "most-wrong",
        "reason": entry.get("reason") or {},
        "labeled": t in live,
        "reviewed": bool(live.get(t, False)),
    }


def _number(value) -> float | None:
    """``value`` as a finite float, or ``None`` -- so a bad field cannot reach the client.

    NaN/inf are ``None`` too: they are not valid JSON numbers, and a bare ``NaN`` token
    in the response body would fail the browser's own ``JSON.parse``.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _read_suggestions(path: Path | None) -> dict | None:
    """The parsed sidecar at ``path``, or ``None`` when there is nothing to show.

    Thin wrapper over :func:`deeperfly.acquisition.read_suggestions`, which already
    returns ``None`` for a file that is absent, unreadable, not a JSON object, or
    stamped with a format version it does not know. The extra guard here is only that
    an unexpected exception from the reader must not take the editor down with it: "no
    suggestions" is a state the panel renders gracefully, whereas a 500 in the sidebar
    would look like a broken editor.
    """
    if path is None:
        return None
    try:
        return read_suggestions(path)
    except (
        Exception
    ) as exc:  # pragma: no cover -- the reader is contracted not to raise
        log.warning("ignoring unreadable suggestions sidecar %s: %s", path, exc)
        return None


def _suggestions_stale(session: Session, data: dict, *, n_done: int) -> dict:
    """How out of date the queue is, as ``{"level", "reasons"}``.

    The two tiers that need to *inspect files* are delegated to
    :func:`deeperfly.acquisition.suggestions_staleness`, so they are decided in one place:
    the recording fingerprint (``hard`` -- the queue's frame indices mean something else
    entirely, and the panel then renders no rows at all) and ``results.h5``'s
    stat-then-md5 fingerprint (``predictions`` -- the ranking describes predictions that
    no longer exist).

    ``progress`` -- some queued frames now carry human work -- is decided here from
    ``n_done`` and carries no reason string, deliberately: the panel recomputes that count
    live from the frames it has just edited (a row flips the moment its first point is
    dragged, with no refetch), so a sentence composed here would be a second, staler copy
    of one the operator is already reading. It is the *expected* state of a queue being
    worked through, which is why it never suppresses the rows.
    """
    stale = suggestions_staleness(
        data,
        identity=session.identity or None,
        results_path=session.results_path,
    )
    level = stale["level"]
    if level == "none" and n_done:
        level = "progress"
    return {"level": level, "reasons": list(stale["reasons"])}


def _suggestions_notes(data: dict, *, n_out_of_range: int) -> list[str]:
    """Caveats about the queue worth putting on screen, not only in the CLI log.

    Each line exists because reading the list without it invites a wrong conclusion: a
    short queue looks like the top-N when the spacing constraint actually ran out of
    slots; a reseeded result's *stored* reprojection error is ~0 on the contralateral
    cells by construction, so the note records that the pristine detector array was
    scored instead; cameras that never went through bundle adjustment (or a recording
    whose median disagreement already exceeds the threshold) mean the ranking may be
    tracking calibration error rather than the detector's mistakes.
    """
    notes: list[str] = []
    shortfall = data.get("shortfall") or {}
    requested, selected = shortfall.get("requested"), shortfall.get("selected")
    if requested and selected is not None and selected < requested:
        why = shortfall.get("reason")
        notes.append(
            f"{selected} of {requested} requested" + (f" — {why}" if why else "")
        )
    source = data.get("source") or {}
    if source.get("reseeded"):
        notes.append(
            "contralateral 2D reseeded from the 3D, so the queue scored "
            f"{source.get('scored_array') or 'pose2d/points'} — the stored reprojection "
            "error is ~0 there and would have ranked nothing"
        )
    cameras_from = source.get("cameras_from")
    if cameras_from and cameras_from != "bundle_adjustment":
        notes.append(
            f"scored against the {cameras_from} cameras (no bundle adjustment) — a high "
            "score may be calibration error, not a detector mistake"
        )
    median = (data.get("coverage") or {}).get("global_residual_median_px")
    threshold = (data.get("params") or {}).get("threshold_px")
    if median is not None and threshold and median > threshold:
        notes.append(
            f"the recording's median disagreement ({median:.1f} px) already exceeds the "
            f"{threshold:g} px threshold — the ranking may be tracking calibration error"
        )
    if n_out_of_range:
        notes.append(
            f"{n_out_of_range} suggested frame(s) fall outside this recording and were "
            "dropped"
        )
    return notes


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
    verbose = bool(msg.get("verbose", False))
    typ = msg.get("type")
    # A live 3D drag (edit_3d with fix=False) streams one edit per animation frame.
    # The NMF overlay reprojection needs a per-frame IK re-fit, so recompute it only
    # on the pin/settle reply -- not ~60x/s mid-drag (the dominant source of drag lag).
    live_drag = typ == "edit_3d" and not bool(msg.get("fix", False))
    # undo/redo can revert an edit on a *different* frame than the one being viewed;
    # `goto` tells the client to navigate there before applying the repaint.
    goto: int | None = None
    notice: str | None = None
    if typ == "edit_2d":
        notice = s.absent_refusal(int(msg["point"]), t)
        s.apply_2d_edit(int(msg["view"]), int(msg["point"]), _xy(msg), t)
    elif typ == "edit_3d":
        s.apply_3d_edit(
            int(msg["view"]),
            int(msg["point"]),
            _xy(msg),
            t,
            fix=bool(msg.get("fix", False)),
        )
    elif typ == "set_gt":
        s.set_gt(int(msg["view"]), int(msg["point"]), _xy(msg), t)
    elif typ == "clear_gt":
        s.clear_gt(int(msg["view"]), int(msg["point"]), t)
    elif typ in ("toggle_fixed", "confirm_point"):
        s.toggle_fixed(int(msg["view"]), int(msg["point"]), t)
    elif typ in ("toggle_invisible", "toggle_occluded"):
        notice = s.absent_refusal(int(msg["point"]), t)
        s.toggle_invisible(int(msg["view"]), int(msg["point"]), t)
    elif typ == "confirm":
        targets = [(int(a), int(b)) for a, b in msg.get("targets", [])]
        s.confirm(targets, t)
    elif typ == "reset":
        targets = [(int(a), int(b)) for a, b in msg.get("targets", [])]
        s.reset_targets(targets, t)
    elif typ == "occlude":
        targets = [(int(a), int(b)) for a, b in msg.get("targets", [])]
        s.occlude_targets(targets, t)
    elif typ == "set_nongt_display":
        # Where a non-GT joint of the instance is drawn. Server-side because the server is
        # what resolves the position; it changes no label, so it records no undo step.
        want = str(msg.get("value", "reprojection"))
        if want not in ("reprojection", "seed"):
            notice = f"unknown non-GT display mode {want!r}"
        else:
            s.nongt_display = want
    elif typ == "create_instance":
        # Double-clicking the detected skeleton. The drag paths create one implicitly too,
        # so this is for starting a frame deliberately without authoring anything yet.
        seed_mode = str(msg.get("seed_mode", "triangulate"))
        if seed_mode not in ("triangulate", "copy"):
            notice = f"unknown seeding mode {seed_mode!r}"
        elif not s.create_instance(t, mode=seed_mode):
            notice = "this frame already has an annotation skeleton"
    elif typ == "reseed_instance":
        seed_mode = str(msg.get("seed_mode", "triangulate"))
        if seed_mode not in ("triangulate", "copy"):
            notice = f"unknown seeding mode {seed_mode!r}"
        elif not s.reseed_instance(t, mode=seed_mode):
            notice = "no annotation skeleton in this frame yet"
    elif typ == "clear_gt_targets":
        # Delete the operator's pixels and nothing else -- distinct from "reset", which
        # also lifts an exclusion (see EditorState.clear_gt_targets).
        targets = [(int(a), int(b)) for a, b in msg.get("targets", [])]
        s.clear_gt_targets(targets, t)
    elif typ == "toggle_exclude":
        targets = [(int(a), int(b)) for a, b in msg.get("targets", [])]
        if s.toggle_exclude_targets(targets, t) is None:
            notice = "nothing to mark -- these keypoints are not on this animal"
    elif typ == "undo":
        goto = s.undo()
    elif typ == "redo":
        goto = s.redo()
    elif typ == "reset_point":
        s.reset_point(int(msg["point"]), t)
    elif typ == "reset_point_view":
        s.reset_point_view(int(msg["view"]), int(msg["point"]), t)
    elif typ == "reset_frame":
        s.reset_frame(t)
    elif typ == "set_landmark":
        # A landmark is authored exactly like a keypoint -- click a pixel in one view -- but
        # into its own namespace, and it drives only the rig solve.
        if not s.set_landmark(int(msg["view"]), int(msg["landmark"]), _xy(msg), t):
            notice = "this project declares no calibration landmarks"
    elif typ == "clear_landmark":
        s.clear_landmark(int(msg["view"]), int(msg["landmark"]), t)
    elif typ == "set_reviewed":
        s.set_reviewed(bool(msg["reviewed"]), t)
    elif typ == "set_absent":
        # View-independent, so the batched (view, point) targets collapse to a point set:
        # "absent in view rf but present in lf" is not an expressible state. Frames are a
        # different matter -- a leg can be lost part-way through -- so the client says
        # which scope it means.
        points = sorted({int(b) for _, b in msg.get("targets", [])})
        if "point" in msg:
            points = sorted(set(points) | {int(msg["point"])})
        whole = str(msg.get("scope", "frame")) == "recording"
        want = msg.get("absent")
        cur = s.labels.absent_all_frames() if whole else s.labels.absent_at(t)
        value = (
            bool(want)
            if want is not None
            else not all(bool(cur[p]) for p in points)  # toggle: set unless all are set
        )
        changed = s.set_absent(points, value, t, whole_recording=whole)
        if changed:
            names = ", ".join(s.point_name(p) for p in changed)
            where = f"all {s.n_frames} frames" if whole else f"frame {t}"
            notice = (
                f"{names} marked absent (not on this animal) in {where}, all views"
                if value
                else f"{names} no longer marked absent in {where}"
            )
        goto = None
    else:  # pragma: no cover -- an unknown type is a client bug; ignore it
        log.warning("ignoring unknown edit message type %r", typ)
    reply_frame = goto if goto is not None else t
    payload = _points_payload(
        session, reply_frame, mode, include_nmf=not live_drag, verbose=verbose
    )
    # Echo the client's monotonic edit seq (when present) so the front-end can drop a
    # superseded reply -- a mid-drag re-solve that lands after release would otherwise
    # repaint the joint to a stale position (the snap-back). `goto` (undo/redo target
    # frame) is null for in-place edits.
    payload["seq"] = msg.get("seq")
    payload["goto"] = goto
    if notice:
        payload["notice"] = notice
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
