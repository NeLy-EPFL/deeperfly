"""Web viewer/corrector for deeperfly results (``deeperfly gui``).

FastAPI + uvicorn are core deps, but importing :mod:`deeperfly.gui` only pulls in
the dependency-free core -- :class:`~deeperfly.gui.state.EditorState`, the
corrections sidecar, footage resolution and the
:class:`~deeperfly.gui.session.Session`. :func:`serve` imports FastAPI/uvicorn
lazily so the web stack is loaded only when the ``gui`` command actually runs,
keeping startup cheap for every other command.

The GUI is a browser app: a local server (FastAPI) serves the result's frames
and 2D overlays to a canvas front-end and applies edits over a WebSocket. It
shows every camera view with its 2D skeleton overlay and lets keypoints be
dragged. Corrections are written to a ``corrections.h5`` sidecar and never
overwrite ``results.h5``. In *Edit 3D* mode the triangulated points are
reprojected into each view; dragging one re-solves the 3D point and every other
view's reprojection updates live. Because it is a web app it runs headless and
can be reached from another machine's browser (default-bound to localhost; tunnel
with ``ssh -L`` for remote correction).
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import threading
import time
import webbrowser
from collections.abc import Iterator
from pathlib import Path

from ..results import PoseResult, StageStore
from .corrections import Corrections, load_corrections, save_corrections
from .readers import FrameSource, resolve_camera_files, resolve_footage
from .session import Session
from .state import EditMode, EditorState

__all__ = [
    "EditMode",
    "EditorState",
    "Corrections",
    "load_corrections",
    "save_corrections",
    "FrameSource",
    "resolve_footage",
    "resolve_camera_files",
    "Session",
    "build_session",
    "serve",
]

log = logging.getLogger("deeperfly")

_GUI_IMPORT_HINT = (
    "the deeperfly web GUI failed to import its server stack (FastAPI + uvicorn); "
    "these are core dependencies, so the install looks incomplete -- try "
    "reinstalling deeperfly (`pip install --force-reinstall deeperfly`)"
)


def build_session(
    results_path: str | Path, footage_dir: str | Path | None = None
) -> Session:
    """Load a ``results.h5`` into an editing :class:`Session` (no web deps).

    Loads the result, resolves each camera's footage (from the paths recorded in
    ``results.h5``, then ``footage_dir``), loads any existing ``corrections.h5``
    sidecar, and assembles the :class:`Session`. Cameras whose footage cannot be
    found fall back to blank frames (logged), so the overlays still draw.

    Parameters
    ----------
    results_path
        Path to a ``results.h5`` file.
    footage_dir
        Optional directory to search for the footage when the recorded paths no
        longer resolve.

    Returns
    -------
    Session
        The assembled session, ready to hand to :func:`serve` /
        :func:`~deeperfly.gui.server.create_app`.
    """
    results_path = Path(results_path)
    result = PoseResult.load(results_path)
    store = StageStore(results_path)
    footage = store.read_footage()
    image_sizes = store.read_image_sizes()
    n_points = int(result.pts2d.shape[2])
    results_dir = results_path.parent
    corrections_path = results_dir / "corrections.h5"

    resolved, missing = resolve_footage(footage, results_dir, footage_dir)
    if not footage:
        log.warning(
            "results.h5 records no footage paths; showing overlays on blank frames "
            "-- re-run 'deeperfly run' to embed them"
        )
    if missing:
        log.warning(
            "footage not found for %s (blank frames); pass --footage-dir to point at it",
            ", ".join(missing),
        )

    source = FrameSource(resolved, image_sizes=image_sizes)

    corrections = load_corrections(
        corrections_path, result.n_views, result.n_frames, n_points
    )
    state = EditorState.from_result(result, corrections)
    return Session.build(
        state,
        source,
        results_path=str(results_path),
        corrections_path=corrections_path,
        image_sizes=image_sizes,
    )


def serve(
    results_path: str | Path,
    footage_dir: str | Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    exit_on_close: bool = True,
) -> None:
    """Open the viewer/corrector on a ``results.h5`` and run the web server.

    Builds the session (:func:`build_session`), starts the FastAPI app under
    uvicorn, and (unless ``open_browser`` is false) opens a browser at the URL
    once the server is accepting connections. Blocks until the server stops.

    Parameters
    ----------
    results_path
        Path to a ``results.h5`` file.
    footage_dir
        Optional directory to search for the footage if the recorded paths no
        longer resolve.
    host
        Address to bind. The loopback default keeps the editor private; bind a
        routable address (e.g. ``0.0.0.0``) only behind a trusted network -- it
        is unauthenticated. Prefer an ``ssh -L`` tunnel for remote correction.
    port
        TCP port to bind; ``0`` picks a free one.
    open_browser
        Whether to open a browser at the served URL on startup.
    exit_on_close
        Stop the server once the last browser tab closes (a brief grace after its
        socket drops, so a refresh can reconnect first). Set false to keep it
        running across tab closes (reconnect later or stop with the Close button /
        Ctrl+C).

    Raises
    ------
    ImportError
        If the web stack (FastAPI + uvicorn) cannot be imported -- these are
        core dependencies, so this signals an incomplete install.
    """
    session = build_session(results_path, footage_dir)
    try:
        import uvicorn

        from .server import create_app
    except ImportError as exc:
        raise ImportError(_GUI_IMPORT_HINT) from exc

    if port == 0:
        port = _free_port(host)

    # The GUI's Close button POSTs /api/shutdown, which calls this to stop the
    # server: flipping should_exit lets uvicorn finish the in-flight reply, then
    # its run loop returns and serve() unblocks (the same as a Ctrl+C). The name
    # `server` is bound below, before any request can trigger this.
    def request_shutdown() -> None:
        server.should_exit = True

    app = create_app(
        session, on_shutdown=request_shutdown, exit_on_disconnect=exit_on_close
    )

    display_host = (
        "localhost" if host in ("0.0.0.0", "127.0.0.1", "::", "::1") else host
    )
    url = f"http://{display_host}:{port}/"
    log.info("deeperfly gui serving %s at %s", session.results_path, url)
    if open_browser:
        connect_host = "127.0.0.1" if host in ("0.0.0.0", "::") else display_host
        threading.Thread(
            target=_open_when_ready, args=(url, connect_host, port), daemon=True
        ).start()

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


def _free_port(host: str) -> int:
    """Pick a free TCP port on ``host`` (bind to 0, read it back, release)."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _open_when_ready(url: str, host: str, port: int, *, timeout: float = 15.0) -> None:
    """Open ``url`` in a browser once the server is accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                break
        except OSError:
            time.sleep(0.1)
    # The launcher webbrowser shells out to may be an external wrapper that
    # inherits our stderr and leaks its own diagnostics into the otherwise clean
    # CLI output -- e.g. VS Code's `browser.sh` runs Node, which prints a
    # `url.parse()` deprecation warning. The child has no use for our stdio, so
    # silence it for the launch.
    with _quiet_child_output():
        webbrowser.open(url)


@contextlib.contextmanager
def _quiet_child_output() -> Iterator[None]:
    """Redirect OS-level stdout/stderr to ``os.devnull`` for the block.

    webbrowser offers no hook to redirect the browser process it spawns, so we
    redirect the file descriptors around the launch and restore them after. This
    runs on a dedicated startup thread for the brief, one-shot browser launch, so
    the global redirection costs nothing -- nothing else is writing to the
    terminal in that window. Python's logging handlers keep their `sys.stderr`
    stream (fd 2), so logging resumes intact once the descriptors are restored.
    """
    with open(os.devnull, "w") as devnull:
        saved_out, saved_err = os.dup(1), os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
