"""The FastAPI app: serves frames + 2D overlays and applies edits over a socket.

:func:`create_app` wraps a :class:`~deeperfly.gui.session.Session` (the Qt-free
editor model) in HTTP + WebSocket handlers: it builds the
:class:`~deeperfly.gui.appstate.EditorApp` that holds the server's state and mounts the
endpoints defined in :mod:`deeperfly.gui.routes`. The browser front-end (``web/``)
fetches metadata and per-frame overlays as JSON, pulls each camera's frame as a
JPEG, and streams edits over ``/ws`` -- every edit maps one-to-one onto an
:class:`~deeperfly.gui.state.EditorState` method and replies with the refreshed
per-view points so the canvases repaint (the same flow the old Qt window drove
with signals). Edits live only in memory until ``POST /api/save`` writes
the ``labels.h5`` sidecar.

Unsaved work is **project-wide**, not per recording: the server keeps every
session the operator has opened this run (:attr:`~deeperfly.gui.appstate.EditorApp.opened`), so switching
recordings costs nothing and loses nothing. ``POST /api/save`` writes the open
recording, ``POST /api/save-all`` writes every recording still holding unsaved labels,
and the only moment an operator has to be asked about it is closing the editor.

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

import logging
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from .appstate import _WEB_DIR, EditorApp
from .routes import router
from .session import Session

__all__ = ["create_app"]

log = logging.getLogger("deeperfly")


def create_app(
    session: Session,
    *,
    on_shutdown: Callable[[], None] | None = None,
    exit_on_disconnect: bool = False,
    disconnect_grace: float = 0.5,
    jobs=None,
) -> FastAPI:
    """Build the FastAPI app serving and editing ``session``.

    ``jobs``, when given, is a :class:`~deeperfly.project.jobs.JobQueue` the browser may submit
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
    app.state.editor = EditorApp(
        session=session,
        on_shutdown=on_shutdown,
        exit_on_disconnect=exit_on_disconnect,
        disconnect_grace=disconnect_grace,
        jobs=jobs,
    )

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

    app.include_router(router)
    return app
