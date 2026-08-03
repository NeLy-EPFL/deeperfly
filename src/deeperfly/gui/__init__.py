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

from ..acquisition import SUGGESTIONS_FILENAME
from ..results import PoseResult, StageStore
from .corrections import Corrections, load_corrections, save_corrections
from .labels import (
    Labels,
    labels_identity,
    load_labels,
    migrate_from_corrections,
    save_labels,
)
from .readers import FrameSource, resolve_camera_files, resolve_footage
from .session import Session
from .state import EditMode, EditorState

__all__ = [
    "EditMode",
    "EditorState",
    "Labels",
    "load_labels",
    "save_labels",
    "labels_identity",
    "migrate_from_corrections",
    "Corrections",
    "load_corrections",
    "save_corrections",
    "FrameSource",
    "resolve_footage",
    "resolve_camera_files",
    "Session",
    "build_session",
    "build_uncalibrated_session",
    "open_target",
    "serve",
]

log = logging.getLogger("deeperfly")

_GUI_IMPORT_HINT = (
    "the deeperfly web GUI failed to import its server stack (FastAPI + uvicorn); "
    "these are core dependencies, so the install looks incomplete -- try "
    "reinstalling deeperfly (`pip install --force-reinstall deeperfly`)"
)


def build_uncalibrated_session(
    project, entry, *, footage_dir: str | Path | None = None
) -> Session:
    """An editing session for a project recording that has never been run.

    The from-scratch case: footage exists, no detector has run, no rig has been solved.
    The session carries an all-NaN :meth:`~deeperfly.results.PoseResult.uncalibrated`
    result, so the editor's derived-3D machinery stays inert and every view is an
    independent 2D canvas -- which is what the operator is there to fill in.

    Frame count comes from the footage itself (there is no ``results.h5`` to ask), and
    the view names come from the recording's ``recording.toml`` footage table, so the
    canvases are labelled with the operator's own camera names rather than ``view0``.

    Parameters
    ----------
    project
        The :class:`~deeperfly.project.Project`.
    entry
        The :class:`~deeperfly.project.RecordingEntry` to open.
    footage_dir
        Optional directory to search when the recorded footage paths no longer resolve.

    Returns
    -------
    Session
        The session, ready to serve.

    Raises
    ------
    SystemExit
        If the recording's footage cannot be resolved at all -- with nothing to show and
        no predictions to fall back on, an editor would be a set of blank rectangles.
    """
    footage = _recording_footage(project, entry)
    if not footage:
        raise SystemExit(
            f"recording {entry.slug!r} has no resolvable footage, and no results.h5 to "
            "fall back on -- there would be nothing to label. Check the paths in "
            f"{project.recording_dir(entry) / 'recording.toml'}, or re-add the recording"
        )
    # Reuse the same resolver the calibrated path uses, in its `{"abs": [...]}` shape,
    # so the absolute-then-by-name fallback (and --footage-dir) behaves identically.
    resolved_or_none = {
        name: resolve_camera_files(
            {"abs": [str(p) for p in paths]},
            project.recording_dir(entry),
            footage_dir,
        )
        for name, paths in footage.items()
    }
    resolved = {name: files for name, files in resolved_or_none.items() if files}
    if not resolved:
        raise SystemExit(
            f"none of {entry.slug!r}'s footage files could be found (looked beside "
            f"{project.recording_dir(entry)}"
            + (f" and in {footage_dir}" if footage_dir else "")
            + ") -- pass --footage-dir"
        )

    source = FrameSource(resolved)
    n_frames = source.n_frames()
    if not n_frames:
        raise SystemExit(f"could not read a frame count from {entry.slug!r}'s footage")
    view_names = list(resolved)
    image_sizes = _probe_image_sizes(source, view_names)
    result = PoseResult.uncalibrated(
        project.skeleton(),
        n_views=len(view_names),
        n_frames=int(n_frames),
        view_names=view_names,
    )
    labels_path = project.outputs_dir(entry) / "labels.h5"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    identity = labels_identity(
        point_names=list(result.skeleton.point_names),
        camera_names=view_names,
        n_frames=int(n_frames),
        image_sizes=image_sizes,
        footage={
            name: {"abs": [str(p) for p in files]} for name, files in resolved.items()
        },
    )
    labels = load_labels(labels_path, identity=identity)
    state = EditorState.from_result(result, labels, image_sizes=image_sizes)
    log.info(
        "uncalibrated session: %d view(s), %d frame(s), no rig -- every view is an "
        "independent 2D canvas until a calibration is solved",
        len(view_names),
        n_frames,
    )
    # The index cached `n_frames = None` (adoption had no results.h5 to ask). Opening the
    # footage is the first time anyone knows, so record it -- otherwise `project status`
    # reports "?" for the whole from-scratch phase.
    if entry.n_frames != int(n_frames):
        try:
            project.update_recording(entry.id, n_frames=int(n_frames))
        except Exception:  # a read-only project must not block editing
            log.debug("could not backfill the frame count into the project index")
    return Session.build(
        state,
        source,
        results_path=str(project.results_path(entry)),
        labels_path=labels_path,
        identity=identity,
        image_sizes=image_sizes,
        project_root=project.root,
        recording_slug=entry.slug,
    )


def _recording_footage(project, entry) -> dict[str, list[Path]]:
    """``camera -> footage paths`` for a project recording, from its ``recording.toml``.

    Falls back to re-discovering the videos beside the recorded origin, so a recording
    adopted before the footage table existed still opens.
    """
    import tomllib

    path = project.recording_dir(entry) / "recording.toml"
    if path.exists():
        table = tomllib.loads(path.read_text()).get("recording", {})
        footage = table.get("footage") or {}
        out = {
            name: [Path(p) for p in (spec.get("abs") or spec.get("names") or [])]
            for name, spec in footage.items()
            if isinstance(spec, dict)
        }
        if any(out.values()):
            return out
    origin = (entry.origin or {}).get("from")
    if origin and Path(origin).is_dir():
        from ..project import discover_footage

        return discover_footage(Path(origin))
    return {}


def _probe_image_sizes(source: "FrameSource", names: list[str]) -> dict:
    """``camera -> (height, width)`` by decoding one frame per view.

    An uncalibrated recording has no ``results.h5`` to have recorded these, and the
    labels identity pins the pixel space its GT lives in -- so they have to be measured,
    once, at open time.
    """
    sizes: dict[str, tuple[int, int]] = {}
    for name in names:
        frame = source.frame(name, 0)
        if frame is not None:
            sizes[name] = (int(frame.shape[0]), int(frame.shape[1]))
    return sizes


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
    results_dir = results_path.parent
    labels_path = results_dir / "labels.h5"
    # The frame-suggestion queue `deeperfly labels-suggest` writes here, if it has been
    # run. Read-only for the GUI, and optional: absent just means an empty Suggested tab.
    suggestions_path = results_dir / SUGGESTIONS_FILENAME

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

    identity = labels_identity(
        point_names=list(result.skeleton.point_names),
        camera_names=list(result.cameras.names),
        n_frames=result.n_frames,
        image_sizes=image_sizes,
        footage=footage,
    )
    labels = _load_or_migrate_labels(labels_path, results_dir, result, identity)
    mesh_hide, template, articulation, ann, tri = _ik_config(results_dir)
    # The pristine detections (result.pts2d is the triangulation-*cleaned* array, so a
    # rejected point is NaN there); they seed a placeholder for an otherwise-unobserved
    # joint so it can still be dragged into a GT label.
    raw = store.read_pose2d()
    state = EditorState.from_result(
        result,
        labels,
        ann=ann,
        tri=tri,
        template=template,
        articulation=articulation,
        raw_pts2d=None if raw is None else raw[0],
        image_sizes=image_sizes,
    )
    return Session.build(
        state,
        source,
        results_path=str(results_path),
        labels_path=labels_path,
        suggestions_path=suggestions_path,
        identity=identity,
        footage=footage,
        image_sizes=image_sizes,
        nmf_hide_parts=mesh_hide,
    )


def _load_or_migrate_labels(labels_path, results_dir, result, identity):
    """Load ``labels.h5`` if present, else migrate a legacy ``corrections.h5`` (if any).

    A one-time migration keeps existing manual corrections usable: the legacy dense
    sidecar is converted to sparse labels in memory (its file is left untouched), and
    the lossy drops are logged. Returns ``None`` (an empty overlay) when neither
    sidecar exists.
    """
    labels = load_labels(labels_path, identity=identity)
    if labels is not None:
        return labels
    corrections_path = results_dir / "corrections.h5"
    corrections = load_corrections(
        corrections_path, result.n_views, result.n_frames, int(result.pts2d.shape[2])
    )
    if corrections is None:
        return None
    labels, report = migrate_from_corrections(corrections, result.pts2d)
    log.warning(
        "migrated %s -> sparse labels (%d GT, %d occluded; dropped %d ambiguous "
        "occlusion(s), %d pure-3D edit(s)); saving writes %s",
        corrections_path.name,
        report["gt"],
        report["occluded"],
        report["dropped_ambiguous_occluded"],
        report["dropped_pts3d_only"],
        labels_path.name,
    )
    labels.dirty = True  # so the operator is prompted to persist the migrated labels
    return labels


def _ik_config(results_dir: Path):
    """Overlay + IK-model + annotation settings from the run config beside ``results.h5``.

    Returns ``(mesh_hide, template, articulation, annotation, triangulation)``: the
    ``[gui].mesh_hide`` overlay parts to hide (default ``["wings"]``); the kinematic
    template + head/abdomen articulation the pipeline fit -- so the editor's live
    re-fit uses the **same** model (restricted legs, custom bounds,
    ``fit_head``/``fit_abdomen``, and custom marker placement all carry over); and the
    ``[annotation]`` solve policy + shared ``[triangulation]`` params so the editor's
    live 3D matches the run. ``template`` / ``articulation`` are ``None`` when no config
    snapshot is present (a bare ``results.h5``), in which case the live re-fit falls
    back to the packaged NeuroMechFly model. ``annotation`` / ``triangulation`` fall
    back to their packaged defaults.
    """
    from ..config import AnnotationParams, Config, TriangulationParams
    from ..inverse_kinematics.articulation import Articulation

    config_path = results_dir / "config.toml"
    if not config_path.exists():
        return ["wings"], None, None, AnnotationParams(), TriangulationParams()
    try:
        config = Config.from_toml(config_path)
    except Exception:  # a malformed snapshot should not block the editor
        log.warning(
            "could not read the run config %s; using overlay defaults", config_path
        )
        return ["wings"], None, None, AnnotationParams(), TriangulationParams()

    mesh_hide = ["wings"]
    template = articulation = None
    annotation, triangulation = AnnotationParams(), TriangulationParams()
    try:
        mesh_hide = list(config.gui.mesh_hide)
    except Exception:
        log.warning(
            "could not read [gui].mesh_hide from %s; hiding the wings", config_path
        )
    try:
        annotation = config.annotation
        triangulation = config.triangulation
    except Exception:
        log.warning(
            "could not read [annotation]/[triangulation] from %s; using defaults",
            config_path,
        )
    try:
        template = config.ik_template()
        articulation = config.ik_articulation() or Articulation.load(fit=())
    except Exception:
        log.warning(
            "could not build the IK model from %s; the live overlay uses the packaged "
            "NeuroMechFly model",
            config_path,
        )
        template = articulation = None
    return mesh_hide, template, articulation, annotation, triangulation


def open_target(
    path: str | Path, *, recording: str | None = None, footage_dir=None
) -> Session:
    """Build a session for whatever the operator pointed at.

    Three things resolve here, so the CLI does not have to know the difference:

    - a **project** directory -- opens one of its recordings (``recording`` names which;
      a project with exactly one needs no name);
    - a ``results.h5`` or a directory holding one -- the pre-existing behavior, unchanged;
    - a project recording that has **never been run** -- an uncalibrated session
      (:func:`build_uncalibrated_session`), which is the from-scratch path.

    Parameters
    ----------
    path
        A project directory, a ``results.h5``, or a directory containing one.
    recording
        Which recording to open when ``path`` is a project (slug, id, or id prefix).
    footage_dir
        Optional directory to search for footage.

    Returns
    -------
    Session
        The assembled session.

    Raises
    ------
    SystemExit
        If a project holds several recordings and none was named, or the named one does
        not exist.
    """
    from ..project import PROJECT_FILENAME, Project

    target = Path(path)
    if not (target / PROJECT_FILENAME).exists():
        if recording:
            log.warning("--recording is only meaningful for a project; ignoring it")
        return build_session(_results_in(target), footage_dir)

    project = Project.load(target)
    if not project.recordings:
        raise SystemExit(
            f"project {project.name!r} has no recordings yet -- "
            f"'deeperfly project add {target} <recording>'"
        )
    if recording is None:
        if len(project.recordings) > 1:
            listing = "\n".join(f"  {e.slug:<40} {e.id}" for e in project.recordings)
            raise SystemExit(
                f"project {project.name!r} holds {len(project.recordings)} recordings; "
                f"name one with --recording:\n{listing}"
            )
        entry = project.recordings[0]
    else:
        try:
            entry = project.recording(recording)
        except KeyError as exc:
            raise SystemExit(str(exc).strip("'")) from None

    results = project.results_path(entry)
    if results.exists():
        session = build_session(results, footage_dir)
        # build_session works from a results.h5 alone (its bare-recording contract), so the
        # project context is attached here rather than threaded through it.
        session.project_root = project.root
        session.recording_slug = entry.slug
        return session
    log.info(
        "%s has no results.h5 -- opening it uncalibrated (2D labeling only)", entry.slug
    )
    return build_uncalibrated_session(project, entry, footage_dir=footage_dir)


def _results_in(target: Path) -> Path:
    """A ``results.h5`` from a file or a directory holding one.

    Raises
    ------
    SystemExit
        With the paths that were tried, since "not found" is otherwise unactionable.
    """
    if target.is_file():
        return target
    for candidate in (
        target / "results.h5",
        target / "deeperfly_outputs" / "results.h5",
    ):
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"no results.h5 at {target} (looked at ./results.h5 and "
        "./deeperfly_outputs/results.h5), and it is not a project directory either"
    )


def serve(
    results_path: str | Path,
    footage_dir: str | Path | None = None,
    *,
    recording: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
    exit_on_close: bool = True,
) -> None:
    """Open the viewer/corrector on a result *or a project* and run the web server.

    Builds the session (:func:`open_target`), starts the FastAPI app under
    uvicorn, and (unless ``open_browser`` is false) opens a browser at the URL
    once the server is accepting connections. Blocks until the server stops.

    Parameters
    ----------
    results_path
        A ``results.h5``, a directory containing one, or a **project** directory.
    recording
        Which recording to open when ``results_path`` is a project.
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
    session = open_target(results_path, recording=recording, footage_dir=footage_dir)
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

    # A job queue only where jobs make sense: they run `deeperfly <subcommand>` in the
    # project directory, so a bare results.h5 session has nothing to run them against and
    # reports `enabled: false` instead.
    queue = None
    if session.project_root is not None:
        from ..jobs import JobQueue

        queue = JobQueue(session.project_root)

    app = create_app(
        session,
        on_shutdown=request_shutdown,
        exit_on_disconnect=exit_on_close,
        jobs=queue,
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
    try:
        server.run()
    finally:
        if queue is not None:
            # Cancel anything still running: a job outliving the editor would keep writing
            # into a results.h5 nobody is watching.
            queue.shutdown()


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
