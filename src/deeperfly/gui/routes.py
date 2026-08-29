"""Every HTTP and WebSocket endpoint the editor serves.

One module-level handler per endpoint, hung on a single :data:`router`. Each reaches the
running server's state through ``editor: EditorApp = Depends(get_editor)`` rather than by
closing over it, which is the whole reason these can be read as a list of endpoints: a
handler's dependencies are in its signature, and the state has one named home
(:mod:`deeperfly.gui.appstate`) instead of twenty variables captured in a closure.

The JSON and image bodies are built in :mod:`deeperfly.gui.payloads`; what is here is
routing, validation, locking and the 409s.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import tomllib
from pathlib import Path

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Response,
    WebSocket,
    WebSocketDisconnect,
)

from ..labels import save_labels
from . import payloads
from .appstate import (
    _UNSAVED_RIG_MSG,
    _WEB_DIR,
    EditorApp,
    _asset_version,
    _open_for_switch,
    _recording_rows,
    get_editor,
)
from .session import Session

log = logging.getLogger("deeperfly")

router = APIRouter()


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
    # The model overlay reprojection needs a per-frame IK re-fit, so recompute it only
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
    elif typ == "set_seed_mode":
        # A session preference, not a per-call argument: the first drag in a frame creates the
        # skeleton too, so a mode that only reached the explicit gesture would not apply on most
        # frames. Changes no label, so it records no undo step.
        want = str(msg.get("value", "triangulate"))
        if want not in ("triangulate", "copy"):
            notice = f"unknown seeding mode {want!r}"
        else:
            s.seed_mode = want
    elif typ == "set_solve_stabilizers":
        # How a point's 3D is derived once its GT views are exclusive: with the unlabelled
        # views filling the direction two GT views cannot see (the default), or from the GT
        # views alone (the older behavior). A session preference like the two above, and it
        # authors no label, so it records no undo step -- but it does change every derived
        # 3D, so the reply must carry a re-solved frame.
        want = str(msg.get("value", "on"))
        if want not in ("on", "off"):
            notice = f"unknown 3D derivation mode {want!r}"
        else:
            s.set_solve_stabilizers(want == "on")
            notice = (
                "3D derived from your pixels alone"
                if want == "off"
                else "3D derived from your pixels, with the other views fixing the depth "
                "they cannot"
            )
    elif typ == "create_instance":
        # Starting a frame deliberately, without authoring anything. The drag paths and the
        # bulk Place both create one implicitly, using the same session seed mode.
        if not s.create_instance(t):
            notice = "this frame already has an annotation skeleton"
    elif typ == "reseed_instance":
        if not s.reseed_instance(t):
            notice = "no annotation skeleton in this frame yet"
        else:
            notice = "seeds laid down again from the detections; ground truth kept"
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
    payload = payloads._points_payload(
        session, reply_frame, mode, include_model=not live_drag, verbose=verbose
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


@router.get("/")
def index() -> Response:
    page = _WEB_DIR / "index.html"
    if not page.is_file():  # pragma: no cover
        raise HTTPException(500, "web/index.html is missing (build the GUI)")
    html = page.read_text(encoding="utf-8").replace("__ASSET_V__", _asset_version())
    return Response(content=html, media_type="text/html")


@router.get("/favicon.ico", include_in_schema=False)
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


@router.get("/apple-touch-icon.png", include_in_schema=False)
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


@router.get("/api/meta")
def meta(response: Response, editor: EditorApp = Depends(get_editor)) -> dict:
    # Never cached. After a recording switch this payload is the only thing telling a
    # reloaded page which recording it is now editing -- and which token to stamp into
    # its frame URLs. A heuristically cached copy would have the new page stamping the
    # PREVIOUS recording's token onto the new recording's pictures, which is exactly
    # the collision `_session_version` exists to prevent.
    response.headers["Cache-Control"] = "no-store"
    return payloads._meta_payload(
        editor.session, editor.cache_v, dirty_recordings=editor.dirty_slugs()
    )


@router.get("/api/schema")
def schema(section: str | None = None) -> dict:
    """The settable config keys, their defaults and their documentation.

    Derived from the ``*Params`` dataclasses (:mod:`deeperfly.config.schema`), never
    from a parallel description -- so a new field appears in the GUI's forms on the
    next reload with no second place to update, and its help text is the prose already
    written for it.

    Sections that have no schema (the detection plan, cameras, the skeleton, video
    specs) are open-ended and are reported as such rather than shown as empty forms.
    """
    from ..config.schema import describe, sections, stage_flags_spec

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
            "default_camera",
            "cameras",
            "skeleton",
            "calibration",
            "pose2d.crops",
            "visualization.videos",
        ],
    }


def _image_cache_control(editor: EditorApp, v: str | None) -> str:
    """Long-lived only for URLs stamped with *this* session's token.

    A stamped URL can never collide with another recording's, so it is safe to
    keep (and ``immutable`` spares the revalidation on scrub-back). An unstamped
    or stale-stamped request is not addressed to a unique picture, so it must not
    enter the cache at all -- see :func:`_session_version`."""
    return "max-age=3600, immutable" if v == editor.cache_v else "no-store"


@router.get("/api/frame/{camera}/{t}")
def frame(
    camera: str, t: int, v: str | None = None, editor: EditorApp = Depends(get_editor)
) -> Response:
    # One atomic read of the session per request. These are `def` handlers, so
    # Starlette runs them in a threadpool where `lock` cannot exclude them, and a
    # recording switch landing between two reads would split one response across two
    # recordings. Every handler below takes the same snapshot.
    s, token = editor.session, editor.cache_v
    key = (token, camera, t)
    data = editor.encoded_frames.get(key)
    if data is None:
        img = s.source.frame(camera, t)
        if img is None:
            raise HTTPException(404, f"no frame for {camera!r} at {t}")
        # Monochrome footage arrives as `(H, W)` and encodes to a one-channel JPEG:
        # a quarter of the CPU, because `_to_bgr` has no channels to reverse and the
        # encoder has one plane instead of three. Barely fewer bytes, though -- JPEG
        # already subsamples this footage's flat chroma away.
        ok, buf = cv2.imencode(".jpg", payloads._to_bgr(img))
        if not ok:  # pragma: no cover -- encoder failure is not expected
            raise HTTPException(500, "frame encoding failed")
        data = buf.tobytes()
        editor.encoded_frames.put(key, data)
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": _image_cache_control(editor, v)},
    )


@router.get("/api/mesh/{camera}/{t}")
def mesh(
    camera: str, t: int, v: str | None = None, editor: EditorApp = Depends(get_editor)
) -> Response:
    """The posed NeuroMechFly mesh for ``camera`` at frame ``t`` as an RGBA PNG.

    404 when the result carries no fitted model (IK off). Rendered on demand and
    memoized per ``(recording, camera, frame)`` so scrubbing back is instant; the
    overlay is heavy enough that re-rendering every scrub would lag.
    """
    s, token = editor.session, editor.cache_v
    if not s.state.has_model:
        raise HTTPException(404, "no inverse-kinematics model to overlay")
    key = (token, camera, _clamp_frame(s, t))
    if key not in editor.mesh_cache:
        png = payloads._render_mesh_png(s, camera, key[2])
        if png is None:
            raise HTTPException(404, f"no mesh overlay for {camera!r} at {t}")
        editor.mesh_cache[key] = png
    return Response(
        content=editor.mesh_cache[key],
        media_type="image/png",
        headers={"Cache-Control": _image_cache_control(editor, v)},
    )


@router.get("/api/model/asset")
def model_asset(editor: EditorApp = Depends(get_editor)) -> Response:
    """The static model mesh topology + per-vertex colors (binary), for the client."""
    data = payloads._model_asset_bytes()
    if data is None or not editor.session.state.has_model:
        raise HTTPException(404, "no inverse-kinematics model to overlay")
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Cache-Control": "max-age=3600"},
    )


@router.get("/api/model/verts/{t}")
async def model_verts(t: int, editor: EditorApp = Depends(get_editor)) -> Response:
    """The posed model vertices, normals + valid-face mask for ``t`` (re-fit from edits)."""
    s = editor.session
    if not s.state.has_model:
        raise HTTPException(404, "no inverse-kinematics model to overlay")
    async with editor.lock:
        data = payloads._model_verts_bytes(s, _clamp_frame(s, t))
    if data is None:
        raise HTTPException(404, f"no mesh overlay at frame {t}")
    return Response(content=data, media_type="application/octet-stream")


@router.get("/api/points/{t}")
def points(
    t: int,
    mode: str = "view",
    verbose: bool = False,
    editor: EditorApp = Depends(get_editor),
) -> dict:
    s = editor.session
    return payloads._points_payload(s, _clamp_frame(s, t), mode, verbose=verbose)


@router.get("/api/scene/{t}")
def scene(t: int, editor: EditorApp = Depends(get_editor)) -> dict:
    s = editor.session
    return payloads._scene_payload(s, _clamp_frame(s, t))


@router.get("/api/corrected")
def corrected(editor: EditorApp = Depends(get_editor)) -> dict:
    """The frames the operator has touched (sorted), each with its reviewed flag.

    Drives the editor's corrected-frames list; the front-end refreshes it after
    edits settle, so it tracks every drag, obscure, reset, and reviewed tick live.
    """
    return {"frames": editor.session.state.corrected_frames()}


@router.get("/api/suggestions")
def suggestions(editor: EditorApp = Depends(get_editor)) -> dict:
    """The ranked "label these next" queue, joined with the live editing state.

    Serves the ``labels_suggest.json`` sidecar (see :func:`_suggestions_payload`)
    -- never recomputes it: the ranking triangulates the whole recording, which
    costs seconds, so it is a CLI artefact and this route is a reader. Always
    ``200``: no sidecar is a normal state (``present: false``), not an error.
    """
    return payloads._suggestions_payload(editor.session)


# -- config ---------------------------------------------------------------
#
# The values come from the project's composed config; a write goes to its PROFILE, which
# holds only what differs from the packaged defaults. So the editor and the CLI change
# the same file the same way, and "reset to default" genuinely removes the key rather
# than restating the default as though someone had chosen it.


@router.get("/api/config")
def get_config(editor: EditorApp = Depends(get_editor)) -> dict:
    """Every describable section's current values, with which ones were actually set."""
    from ..config.schema import effective, sections, stage_flags_spec

    if editor.session.project_root is None:
        return {"enabled": False, "reason": "no project", "sections": {}}
    project = editor.project()
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
        except Exception as exc:  # a malformed section must not blank the whole panel
            out[name] = {"error": str(exc)}
            continue
        out[name] = {
            field: {
                "value": payloads._jsonable(value),
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


@router.post("/api/config")
def set_config(payload: dict, editor: EditorApp = Depends(get_editor)) -> dict:
    """Set one ``{"section", "key", "value"}`` in the project's profile.

    ``value: null`` clears the override. Validated through ``Config``'s own strict
    loader, so the editor cannot store a key a run would reject.
    """
    project = editor.project()
    section, key = str(payload.get("section", "")), str(payload.get("key", ""))
    try:
        path = project.set_profile_key(section, key, payload.get("value"))
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from None
    log.info("set %s.%s in %s", section, key, path.name)
    return {"ok": True, "profile": str(path)}


#: Why the tab is unavailable, or ``None``. A bare ``results.h5`` has nowhere to write a
#: calibration, which is the same reason Settings and Jobs are unavailable there -- so the
#: GET routes answer ``enabled: false`` with this sentence rather than an error status. A
#: 4xx would be logged as a console error by the browser, which is how the editor reports
#: a *fault*, and "you opened a file instead of a project" is not one.
_BA_NO_PROJECT = (
    "This session was opened on a bare results.h5. Bundle adjustment needs a project to "
    "write the calibration into — open one with 'deeperfly gui <project>'."
)


def _ba_run_here(editor: EditorApp) -> dict:
    """The solve state, but only when it belongs to the recording that is open.

    There is one ``ba_run`` per SERVER while the tab reporting it survives a recording
    switch (the editor rebuilds in place, and the pane with it). Unstamped, the last
    solve would be announced under whichever animal happens to be open next: a
    calibration written into flyA's directory read as flyB's, and a per-camera table
    matched against a rig it was never solved on. Another recording's run is reported
    as ``idle`` rather than hidden by a flag, so the pane needs no rule of its own.
    """
    with editor.ba_lock:
        run = dict(editor.ba_run)
    if (
        run.get("state") == "idle"
        or run.get("recording") == editor.session.recording_slug
    ):
        return run
    return {"state": "idle"}


def _ba_context(editor: EditorApp):
    """``(project, camera_names, config)`` for the open session, or a 409."""
    project = editor.project()
    names = list(editor.session.state.camera_names)
    config = None
    try:
        from ..config import Config

        config = Config.from_dict(tomllib.loads(project.compose_config()))
    except Exception as exc:  # a malformed profile must not break the whole tab
        log.warning("could not compose the project config for BA defaults: %s", exc)
    return project, names, config


@router.get("/api/bundle-adjust")
def bundle_adjust_plan(
    max_frames: int | None = None, editor: EditorApp = Depends(get_editor)
) -> dict:
    """The tab's initial state: the fix/free matrix, the settings, and the readiness.

    Everything is defaulted from the project config's ``[bundle_adjustment]`` so the tab
    agrees with what a pipeline run would do, and the readiness block is computed BEFORE
    anything is solved -- a rig the labels cannot determine is reported as refused here
    rather than as a flattering residual afterwards.
    """
    from . import ba

    if editor.session.project_root is None:
        return {"enabled": False, "reason": _BA_NO_PROJECT}
    project, names, config = _ba_context(editor)
    has_rig = bool(editor.session.state.result.has_cameras)
    settings, note = ba.settings_from_config(config, names, has_rig=has_rig)
    if max_frames is not None:
        settings.max_frames = max_frames or None
    obs = ba.gather(
        editor.session.state,
        max_frames=settings.max_frames,
        frame_sampling=settings.frame_sampling,
    )
    pre = ba.preflight(obs, settings, names)
    return {
        "enabled": True,
        "recording": editor.session.recording_slug,
        "cameras": names,
        "params": list(ba.PARAMS),
        "losses": list(ba.LOSSES),
        "samplings": list(ba.SAMPLINGS),
        "has_rig": has_rig,
        "cold_start": not has_rig,
        "provisional": list(editor.session.state.provisional_views),
        "settings": settings.to_json(),
        "defaults_note": note,
        "readiness": pre,
        "point_names": list(editor.session.state.result.skeleton.point_names),
        "run": {k: v for k, v in _ba_run_here(editor).items() if k != "cameras"},
    }


@router.post("/api/bundle-adjust/check")
def bundle_adjust_check(payload: dict, editor: EditorApp = Depends(get_editor)) -> dict:
    """Readiness for a candidate fix/free split, without solving anything.

    Ticking a box changes whether the problem is well posed, so the tab re-asks on every
    change. Cheap by construction: it gathers the labels and counts co-visibility, which
    is the same work the plan route does and nothing more.
    """
    from . import ba

    if editor.session.project_root is None:
        return {"enabled": False, "reason": _BA_NO_PROJECT}
    _project, names, _config = _ba_context(editor)
    try:
        settings = ba.BaSettings.from_json(payload.get("settings") or {}, names)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    obs = ba.gather(
        editor.session.state,
        max_frames=settings.max_frames,
        frame_sampling=settings.frame_sampling,
    )
    return ba.preflight(obs, settings, names)


@router.post("/api/bundle-adjust")
def bundle_adjust_run(payload: dict, editor: EditorApp = Depends(get_editor)) -> dict:
    """Start a solve on a worker thread. Poll ``GET /api/bundle-adjust/run``.

    The whole settings object arrives in one request -- a per-key config route cannot
    express "these cameras fixed, those free, with this loss" atomically, and a half-
    applied split would solve something the operator never asked for.
    """
    from ..rig import store as rigstore
    from . import ba

    project, names, config = _ba_context(editor)
    with editor.ba_lock:
        if editor.ba_run.get("state") == "running":
            # Named, because the operator can be standing somewhere else entirely: one
            # worker serves the whole editor, so the solve holding the slot may belong
            # to a recording whose pane is not even on screen.
            other = editor.ba_run.get("recording")
            where = (
                ""
                if other in (None, editor.session.recording_slug)
                else f" (on {other})"
            )
            raise HTTPException(409, f"a bundle adjustment is already running{where}")
    try:
        settings = ba.BaSettings.from_json(payload.get("settings") or {}, names)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    name = str(payload.get("name") or "from-labels").strip() or "from-labels"
    has_rig = bool(editor.session.state.result.has_cameras)
    obs = ba.gather(
        editor.session.state,
        max_frames=settings.max_frames,
        frame_sampling=settings.frame_sampling,
    )
    pre = ba.preflight(obs, settings, names)
    if not pre["ok"]:
        raise HTTPException(400, pre["problems"][0])

    cameras = editor.session.state.result.cameras if has_rig else None
    intrs = dists = None
    if not has_rig:
        # Cold start: intrinsics are never derived from labels, so they must come from
        # the config's own camera specs (the orbit rig's focal length).
        try:
            from ..rig.cameras import CameraGroup

            prior = CameraGroup.from_config(
                config, image_sizes=editor.session.image_sizes
            )
            intrs = np.stack([prior[n].intr for n in names])
            dists = np.stack([prior[n].dist for n in names])
        except Exception as exc:
            raise HTTPException(
                400,
                "this session has no rig and no intrinsics could be read from the "
                f"project config, so a cold-start solve has nothing to start from: {exc}",
            ) from None

    recording = editor.session.recording_slug
    cal_dir = rigstore.recording_dir(Path(project.root), recording)
    sizes = dict(editor.session.image_sizes)

    def work():
        try:
            report = ba.solve(
                cameras,
                obs,
                settings,
                cold_start=not has_rig,
                intrinsics=intrs,
                dists=dists,
            )
            path = rigstore.save_calibration(
                cal_dir,
                report["cameras"],
                name=name,
                image_sizes=sizes,
                obs=obs,
                settings=settings,
                report=report,
                recording=recording,
            )
            with editor.ba_lock:
                editor.ba_run.clear()
                editor.ba_run.update(
                    state="done",
                    recording=recording,
                    calibration=str(path),
                    calibration_file=path.name,
                    before=report["before"],
                    after=report["after"],
                    moved=report["moved"],
                    fixed_refs=report["fixed_refs"],
                    success=report["success"],
                    nfev=report["nfev"],
                    message=report["message"],
                    readiness=pre,
                    settings=settings.to_json(),
                )
        except Exception as exc:  # reported to the operator, never swallowed
            log.exception("bundle adjustment failed")
            with editor.ba_lock:
                editor.ba_run.clear()
                editor.ba_run.update(
                    state="failed",
                    recording=recording,
                    error=f"{type(exc).__name__}: {exc}",
                )

    with editor.ba_lock:
        editor.ba_run.clear()
        editor.ba_run.update(
            state="running",
            recording=recording,
            name=name,
            n_tracks=obs.n_tracks,
            n_frames=obs.n_frames,
            readiness=pre,
        )
    threading.Thread(target=work, name="deeperfly-ba", daemon=True).start()
    return {"state": "running", "n_tracks": obs.n_tracks, "n_frames": obs.n_frames}


@router.get("/api/bundle-adjust/run")
def bundle_adjust_status(editor: EditorApp = Depends(get_editor)) -> dict:
    return _ba_run_here(editor)


@router.get("/api/calibrations")
def list_calibrations(editor: EditorApp = Depends(get_editor)) -> dict:
    """Every calibration in the project, and which rig the editor is using now."""
    from ..rig import store as rigstore

    if editor.session.project_root is None:
        return {
            "enabled": False,
            "reason": _BA_NO_PROJECT,
            "items": [],
            "active": None,
        }
    project = editor.project()
    # Per recording: a rig belongs to the session it was solved in, and a flat list mixed
    # every recording's calibrations together with no way to tell which applied here.
    cal_dir = rigstore.recording_dir(Path(project.root), editor.session.recording_slug)
    current = getattr(editor.session, "active_calibration", None)
    items = rigstore.list_calibrations(
        cal_dir, active=Path(current) if current else None
    )
    for row in items:
        cams = set(row.get("cameras") or ())
        row["covers_session"] = (
            bool(cams) and set(editor.session.state.camera_names) <= cams
        )
    return {
        "enabled": True,
        "directory": str(cal_dir),
        "active": current,
        "session_cameras": list(editor.session.state.camera_names),
        "items": items,
    }


def _still_provisional(path: Path, current) -> list[str]:
    """Which of ``current`` this calibration did NOT solve -- see :func:`rigstore.still_provisional`.

    The rule lives in ``ba`` so that anything reproducing the editor's derived 3D
    outside the editor makes the identical promotion decision.
    """
    from ..rig import store as rigstore

    return rigstore.still_provisional(path, current)


def _meta_patch(sess, **changes) -> None:
    """Merge ``changes`` into ``results.h5``'s meta, best effort.

    Both the provisional set and the chosen rig live here rather than in the project
    index: they are facts about ONE recording, they must survive a restart, and a bare
    results.h5 session has no project to write to. A read-only file must not break a
    switch, so this logs and carries on.
    """
    try:
        import h5py

        with h5py.File(sess.results_path, "r+") as f:
            meta = json.loads(f.attrs.get("meta", "{}"))
            for key, value in changes.items():
                if value in (None, [], ()):
                    meta.pop(key, None)
                else:
                    meta[key] = value
            f.attrs["meta"] = json.dumps(meta)
    except Exception:
        log.exception("could not record %s in results.h5", ", ".join(changes))


def _restore_active_calibration(sess) -> None:
    """Re-apply the rig this recording was last switched onto.

    Selecting a calibration is a decision about the recording, not about the window that
    happened to be open, so it has to outlive the session -- otherwise every restart
    silently reverts to the rig in results.h5 and the operator's 3D quietly changes
    underneath work they already did against the other one.
    """
    from ..rig import store as rigstore

    path = rigstore.apply_active_calibration(
        sess.state, sess.results_path, sess.project_root, sess.recording_slug
    )
    if path is None:
        return
    sess.active_calibration = str(path)
    log.info("%s: using calibration %s", sess.recording_slug, path.name)


def _base_provisional(sess) -> list[str]:
    """Views the rig in ``results.h5`` never calibrated -- see :func:`rigstore.base_provisional`."""
    from ..rig import store as rigstore

    return rigstore.base_provisional(sess.results_path, sess.state.camera_names)


def _persist_provisional(sess, views) -> None:
    """Record the surviving provisional views in ``results.h5``, best effort.

    Kept on disk so reopening the recording does not silently re-demote a camera that has
    since been solved -- and so nothing has to remember which rig was selected last time.
    """
    try:
        import h5py

        with h5py.File(sess.results_path, "r+") as f:
            meta = json.loads(f.attrs.get("meta", "{}"))
            if list(views):
                meta["provisional_cameras"] = list(views)
            else:
                meta.pop("provisional_cameras", None)
            f.attrs["meta"] = json.dumps(meta)
    except Exception:  # a read-only results.h5 must not fail the switch
        log.exception("could not record the provisional cameras in results.h5")


@router.post("/api/calibrations/delete")
async def delete_calibration(
    payload: dict, editor: EditorApp = Depends(get_editor)
) -> dict:
    """Remove a calibration this editor solved (never the one in use)."""
    from ..rig import store as rigstore

    project = editor.project()
    raw = str(payload.get("calibration") or "")
    if not raw:
        raise HTTPException(400, "no calibration named")
    path = Path(raw)
    if not path.is_absolute():
        path = (
            rigstore.recording_dir(Path(project.root), editor.session.recording_slug)
            / path
        )
    async with editor.lock:
        try:
            rigstore.delete_calibration(
                path, active=getattr(editor.session, "active_calibration", None)
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None
    return {"ok": True, "deleted": path.name}


@router.post("/api/calibrations/select")
async def select_calibration(
    payload: dict, editor: EditorApp = Depends(get_editor)
) -> dict:
    """Derive the editor's non-GT positions from a different calibration.

    Rebinds the rig on the OPEN session: the labels are untouched (they are 2D), but
    every derived 3D is a function of the rig, so the cache of derived positions is
    dropped and re-derived. That cache can hold the operator's only record of a
    hand-placed DEPTH -- a drag on a point with fewer than two usable views stores a
    ray-slide the labels cannot reproduce -- so this refuses while there are unsaved
    labels unless ``discard`` is set, exactly as a recording switch does.
    """
    from ..rig import store as rigstore
    from ..rig.cameras import CameraGroup

    project = editor.project()
    raw = str(payload.get("calibration") or "")
    if editor.session.state.dirty and not bool(payload.get("discard")):
        raise HTTPException(409, _UNSAVED_RIG_MSG)
    names = list(editor.session.state.camera_names)
    if not raw:
        # Back to the rig that came out of the pipeline. Reachable because the choice
        # now persists: without a way back, one click would pin a recording to a
        # calibration for good.
        from ..results import StageStore

        stored = StageStore(Path(editor.session.results_path))
        group = stored.read_cameras("bundle_adjustment") or stored.read_cameras(
            "pose2d"
        )
        if group is None:
            raise HTTPException(409, "this results.h5 carries no rig to fall back to")
        async with editor.lock:
            editor.session.state.result.cameras = group
            editor.session.state.invalidate_derived()
            # Whatever was provisional when the pipeline wrote this file is provisional
            # again: the rig that solved it is no longer the one in use.
            editor.session.state.set_provisional(_base_provisional(editor.session))
            _meta_patch(editor.session, active_calibration=None)
            editor.session.active_calibration = None
            editor.cache_v = payloads._session_version(editor.session)
        log.info("editor rig switched back to the one in results.h5")
        return {
            "ok": True,
            "calibration": None,
            "file": None,
            "cache_v": editor.cache_v,
        }
    path = Path(raw)
    if not path.is_absolute():
        path = (
            rigstore.recording_dir(Path(project.root), editor.session.recording_slug)
            / path
        )
    if not path.is_file():
        raise HTTPException(404, f"no such calibration: {path}")
    async with editor.lock:
        try:
            group = CameraGroup.from_calibration(path, names=names)
        except (ValueError, KeyError) as exc:
            raise HTTPException(
                400,
                f"{path.name} does not cover this session's views {names}: {exc}",
            ) from None
        editor.session.state.result.cameras = group
        # Every derived 3D came from the OLD rig; keeping any of it would mix two
        # geometries in one file. Cleared wholesale, then re-derived on demand.
        editor.session.state.invalidate_derived()
        # A view stops being display-only exactly when a calibration SOLVED it -- not
        # merely when one mentions it. The provenance records what was held fixed, so a
        # rig that pinned the hind camera and refined the others leaves it provisional.
        base = _base_provisional(editor.session)
        still = _still_provisional(path, base)
        if set(still) != set(editor.session.state.provisional_views):
            promoted = sorted(set(editor.session.state.provisional_views) - set(still))
            if promoted:
                log.info(
                    "%s solved %s; it now informs the other views",
                    path.name,
                    promoted,
                )
        editor.session.state.set_provisional(still)
        _meta_patch(editor.session, active_calibration=path.name)
        editor.session.active_calibration = str(path)
        editor.cache_v = payloads._session_version(editor.session)
    log.info("editor rig switched to %s", path.name)
    return {
        "ok": True,
        "calibration": str(path),
        "file": path.name,
        "cache_v": editor.cache_v,
    }


# -- recordings -----------------------------------------------------------
#
# The editor shows exactly one recording; this is how the operator changes which,
# without restarting the server and without hunting for the next results.h5 in a
# shell. The swap happens in this process -- `session` and `cache_v` are rebound
# below -- and every open browser is then told to reload, because the front-end
# builds its canvases, its key bindings and its frame-URL cache token from /api/meta,
# which it fetches exactly once per page load.
#
# Deliberately restricted to the CURRENT project. The JobQueue is built once, from
# `session.project_root`, in `deeperfly.gui.serve`; a cross-project swap would leave
# it running commands in -- and writing logs into -- the previous project, and
# nothing here would notice. Keeping the project fixed makes that correct by
# construction rather than by a second rebind someone has to remember.


@router.get("/api/recordings")
def list_recordings(editor: EditorApp = Depends(get_editor)) -> dict:
    """This project's recordings with their label counts, and which one is open.

    ``enabled: false`` for a bare ``results.h5`` session -- the same convention as
    ``/api/jobs`` and ``/api/config``, so the picker can say *why* it cannot switch
    instead of rendering an unexplained empty menu. This opens every recording's
    ``labels.h5``, which is why it is a route of its own rather than a field of
    ``/api/meta`` (fetched on every page load, including the ones after a switch).
    """
    s = editor.session
    if s.project_root is None:
        return {
            "enabled": False,
            "reason": "this session was opened on a bare results.h5; open a project "
            "to switch between its recordings",
            "recordings": [],
        }
    project = editor.project()
    return {
        "enabled": True,
        "project": project.name,
        "project_root": str(project.root),
        "active": s.recording_slug,
        "dirty": bool(s.state.dirty),
        # Unsaved work is project-wide (the editor keeps every recording it has
        # opened), so "is anything unsaved" is a different question from "is THIS
        # recording unsaved" -- and it is the one the close prompt asks.
        "project_dirty": editor.project_dirty(),
        "dirty_recordings": editor.dirty_slugs(),
        "recordings": _recording_rows(
            project,
            s.recording_slug,
            {slug: sess for slug, (sess, _) in editor.opened.items()},
        ),
    }


@router.post("/api/recordings/open")
async def open_recording(
    payload: dict, editor: EditorApp = Depends(get_editor)
) -> dict:
    """Switch the editor to another recording of this project, in place.

    ``{"recording": "<slug>", "discard": false}``. **Never refused for unsaved
    labels.** The outgoing session is kept in ``opened``, so its labels and its undo
    history are still there when the operator comes back -- unsaved work spans the
    project, and the editor asks about it once, at close time. Passing
    ``discard: true`` is how a caller says the opposite: throw the outgoing
    recording's unsaved labels away now (it is dropped from the registry).

    Reopening a recording still held in the registry rebinds it *without touching
    disk* -- no video headers, no ``results.h5``, no IK model, and the operator's
    selected calibration and undo stack exactly as they left them.

    A recording not in the registry is built completely, in a worker thread, *before*
    anything is rebound -- so one that cannot be opened leaves the current one
    untouched.
    """
    if editor.session.project_root is None:
        raise HTTPException(
            409,
            "this session was opened on a bare results.h5; open a project to switch "
            "between its recordings",
        )
    slug = str(payload.get("recording") or "")
    if not slug:
        raise HTTPException(400, "no recording named")
    if slug == editor.session.recording_slug:
        return {"switched": False, "recording": slug, "reason": "already open"}
    discard = bool(payload.get("discard"))
    if editor.switch_busy:
        raise HTTPException(409, "a recording switch is already under way")

    root = editor.session.project_root
    try:
        entry = editor.project().recording(slug)
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'")) from None
    slug = entry.slug  # accept an id or an id prefix, then speak in slugs

    editor.switch_busy = True
    try:
        # Opened before the build: a tab that closes while we are still reading the
        # new recording's videos must not be mistaken for the last tab going away.
        editor.begin_switch()
        restored = slug in editor.opened
        if restored:
            new_session, new_v = editor.opened[slug]
        else:
            try:
                new_session, new_v = await asyncio.to_thread(
                    _open_for_switch, root, slug
                )
            except SystemExit as exc:
                # `open_target` is written for the CLI and reports "this cannot be
                # opened" by exiting. SystemExit is a BaseException, so it would sail
                # straight past `except Exception` and out of the request handler.
                raise HTTPException(409, f"could not open {slug!r}: {exc}") from None
            except Exception as exc:
                log.exception("could not open recording %s", slug)
                raise HTTPException(409, f"could not open {slug!r}: {exc}") from None

        async with editor.lock:
            outgoing, outgoing_v = editor.session, editor.cache_v
            editor.session = new_session
            if not restored:
                _restore_active_calibration(editor.session)
                new_v = payloads._session_version(editor.session)
            editor.cache_v = new_v
            # Keyed by slug, so a reopen MOVES the entry to the end rather than
            # duplicating it -- and the outgoing session goes back in under the token
            # it was serving with, not a freshly computed one.
            if outgoing.recording_slug is not None:
                if discard:
                    editor.opened.pop(outgoing.recording_slug, None)
                else:
                    editor.opened[outgoing.recording_slug] = (outgoing, outgoing_v)
            editor.opened.pop(slug, None)
            editor.opened[slug] = (editor.session, new_v)
            editor.prune_sessions()
            editor.mesh_cache.clear()
            # The pictures of a recording nobody is looking at are the largest thing a
            # retained session holds -- the decoded frames, and the reference frames an
            # open decoder keeps per camera -- and the cheapest thing to rebuild: the
            # browser refetches them on the way back, through cursors reopened on
            # demand from readers that stayed. The session itself -- labels, undo
            # history, derived 3D -- is what is kept.
            if outgoing is not editor.session:
                outgoing.source.release_cache()
        # A session dropped here (pruned, or discarded) is deliberately NOT closed:
        # `FrameSource.close()` clears the decoded-frame LRU that `FrameSource.frame`
        # reads outside its try block, and a threadpool handler that snapshotted the
        # outgoing session just before the swap may still be inside it. Nothing leaks:
        # the readers hold no OS handle, the cursors that do hold one were just
        # released above (and close only once a read in flight has left), and the last
        # reference going away reclaims the arrays, the cache and the undo stacks.
        editor.begin_switch()  # restart the window: the reloads begin now
        await editor.broadcast(
            {"type": "reload", "reason": "recording", "recording": slug}
        )
        log.info(
            "editor switched to recording %s (%s)",
            slug,
            "restored from memory" if restored else "opened from disk",
        )
        return {
            "switched": True,
            "recording": slug,
            "cache_v": new_v,
            # Whether this recording came back from memory (with its unsaved labels
            # and its undo history) or was read from disk. The front-end says so.
            "restored": restored,
            "dirty_recordings": editor.dirty_slugs(),
        }
    finally:
        editor.switch_busy = False


# -- jobs ----------------------------------------------------------------
#
# Polled by the panel rather than pushed over `/ws`. The socket carries the *editing*
# stream and is single-writer by design; a read-only tab must still see the queue, and
# a broadcast would either bypass that lock or duplicate it. A 2 s poll of a
# handful-of-rows JSON payload is cheaper than the complexity.


@router.get("/api/jobs")
def list_jobs(tail: int = 5, editor: EditorApp = Depends(get_editor)) -> dict:
    """The queue, newest first. ``enabled: false`` when this session has no queue."""
    if editor.jobs is None:
        return {
            "enabled": False,
            "reason": "this session was opened on a bare results.h5; open a project "
            "to run jobs from the editor",
            "jobs": [],
        }
    return {
        "enabled": True,
        "busy": editor.jobs.busy,
        "jobs": [j.as_dict(tail=tail) for j in editor.jobs.list()],
    }


@router.post("/api/jobs")
async def submit_job(payload: dict, editor: EditorApp = Depends(get_editor)) -> dict:
    """Queue a job. ``{"kind": ..., "argv": [...], "label": ..., "recording": ...}``.

    Only allow-listed kinds are accepted (:data:`~deeperfly.project.jobs.JOB_KINDS`) -- the
    editor is reachable over HTTP, and a queue that ran arbitrary argv would be a
    remote shell.
    """
    if editor.jobs is None:
        raise HTTPException(409, "this session has no job queue (open a project)")
    try:
        job = editor.jobs.submit(
            str(payload.get("kind", "")),
            [str(a) for a in (payload.get("argv") or [])],
            label=str(payload.get("label", "")),
            recording=payload.get("recording"),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return job.as_dict()


@router.get("/api/jobs/{job_id}")
def job_detail(
    job_id: str, tail: int = 200, editor: EditorApp = Depends(get_editor)
) -> dict:
    if editor.jobs is None:
        raise HTTPException(409, "this session has no job queue")
    job = editor.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    return job.as_dict(tail=tail)


@router.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str, editor: EditorApp = Depends(get_editor)) -> dict:
    if editor.jobs is None:
        raise HTTPException(409, "this session has no job queue")
    if editor.jobs.get(job_id) is None:
        raise HTTPException(404, f"no job {job_id}")
    return {"cancelled": editor.jobs.cancel(job_id)}


def _write_session(s: Session) -> None:
    """Write one session's labels sidecar (and mirror its absence declaration).

    Takes the session explicitly rather than reading `session`: it is called for
    recordings that are *not* the open one (``/api/save-all``), and the failure it
    must be incapable of is writing one recording's labels to another's path. The
    caller holds the mutation lock.
    """
    save_labels(
        s.labels_path,
        s.state.labels,
        identity=s.identity,
        subject_id=s.state.labels.subject_id,
    )
    # Mirror the absence declaration into results.h5's `animal/` group. That is the
    # seam the pipeline and every results.h5-only consumer read, so a fact authored
    # here reaches a re-run (and the render path) without anyone parsing labels.h5.
    # An in-place patch, so no stage output is touched.
    try:
        from ..results import StageStore

        StageStore(Path(s.results_path)).write_animal(
            absent=s.state.labels.absent_all_frames(),
            subject_id=s.state.labels.subject_id,
        )
    except Exception:  # a read-only results.h5 must not fail the label save
        log.exception("could not mirror the absence declaration into results.h5")


@router.post("/api/save")
async def save(editor: EditorApp = Depends(get_editor)) -> dict:
    """Write the OPEN recording's labels. See ``/api/save-all`` for the rest."""
    async with editor.lock:
        # One snapshot for the whole save. A recording switch takes this same lock,
        # so it cannot interleave -- but reading `session` seven times would make
        # that a property of the lock rather than of this function, and the failure
        # it prevents is writing the NEW recording's labels to the OLD one's path.
        s = editor.session
        _write_session(s)
        return {
            "dirty": s.state.dirty,
            "project_dirty": editor.project_dirty(),
            "dirty_recordings": editor.dirty_slugs(),
        }


@router.post("/api/save-all")
async def save_all(editor: EditorApp = Depends(get_editor)) -> dict:
    """Write every recording still holding unsaved labels -- the editor's Save.

    Unsaved work spans the project (the editor keeps every recording it has opened),
    so "save" means all of it: saving only the open recording would leave the title
    bar starred and the operator hunting for which other one still needs a click.

    A write that fails is *reported*, not raised: the saves that did land must not be
    undone by one unwritable sidecar, and the front-end needs to know it may not
    close. ``failed`` empty and ``project_dirty`` false is the only "everything is on
    disk" answer.
    """
    saved: list[str] = []
    failed: list[dict] = []
    async with editor.lock:
        # A snapshot of the registry: `_write_session` releases nothing, but the list
        # is what makes the set of recordings saved here independent of anything that
        # runs later in this handler.
        targets = [
            (slug, sess)
            for slug, (sess, _) in editor.opened.items()
            if sess.state.dirty
        ]
        if not targets and editor.session.state.dirty:
            # A bare results.h5 -- no slug, no registry entry, still savable.
            targets = [(editor.session.recording_slug or "", editor.session)]
        for slug, sess in targets:
            try:
                _write_session(sess)
            except Exception as exc:  # noqa: BLE001 -- reported, per the docstring
                log.exception("could not save the labels of %s", slug or "session")
                failed.append({"recording": slug, "error": str(exc)})
            else:
                saved.append(slug)
        return {
            "saved": saved,
            "failed": failed,
            "dirty": editor.session.state.dirty,
            "project_dirty": editor.project_dirty(),
            "dirty_recordings": editor.dirty_slugs(),
        }


@router.post("/api/shutdown")
async def shutdown(editor: EditorApp = Depends(get_editor)) -> dict:
    """Stop the server (the GUI's Close button).

    The browser saves or discards any unsaved corrections before calling this,
    so the handler touches no state -- it just signals the run loop to exit.
    Returns first; uvicorn finishes this reply, then shuts down on its next
    tick. A no-op (still ``200``) when no shutdown hook was wired in.
    """
    log.info("gui requested shutdown")
    if editor.on_shutdown is not None:
        editor.on_shutdown()
    return {"ok": True}


@router.websocket("/ws")
async def ws(websocket: WebSocket, editor: EditorApp = Depends(get_editor)) -> None:
    await websocket.accept()
    editor.sockets.add(websocket)
    if editor.pending_exit is not None:
        # A reconnect (typically a page refresh) cancels a pending shutdown.
        editor.pending_exit.cancel()
        editor.pending_exit = None
        log.info("browser reconnected; shutdown cancelled")
    if editor.writer is None:
        # The first (sole) editor. It is told nothing -- an unadorned client is
        # editable by default -- so the single-browser flow (and its tests) is
        # unchanged: the first message it receives is still its own edit reply.
        editor.writer = websocket
    else:
        # A second+ browser: read-only until it takes over or the writer leaves.
        await websocket.send_json(editor.role_msg(websocket))
    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get("type") == "claim":
                # This browser takes over editing (the read-only "Take over"
                # action). The slot is single-valued, so the previous writer,
                # if any, is demoted to read-only.
                old = editor.writer
                editor.writer = websocket
                if old is not None and old is not websocket:
                    await old.send_json(editor.role_msg(old))
                await websocket.send_json(editor.role_msg(websocket))
                continue
            if websocket is not editor.writer:
                # A read-only browser must not mutate the shared session: refuse
                # the edit and re-assert its role (its UI already blocks this, so
                # this is the belt-and-braces server guard).
                log.info("ignoring edit from a read-only browser")
                await websocket.send_json(editor.role_msg(websocket))
                continue
            try:
                async with editor.lock:
                    payload = _handle_edit(editor.session, msg)
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
        editor.sockets.discard(websocket)
        if websocket is editor.writer:
            # The editor left; hand the writer slot to another open browser so a
            # surviving viewer can edit. Left free when no socket remains, so a
            # lone tab's refresh reclaims it on reconnect (writer is None again).
            editor.writer = None
            for cand in list(editor.sockets):
                editor.writer = cand
                try:
                    await cand.send_json(editor.role_msg(cand))
                except Exception:  # pragma: no cover -- the socket is closing
                    editor.writer = None
                    continue
                break
        # The last tab closed: stop the server, but give a refresh's reconnect
        # the grace period to cancel it first. `switching` suppresses this entirely
        # for the duration of a recording switch, where EVERY tab drops its socket at
        # once and the reconnect only comes after the reloaded page has fetched
        # /api/meta -- which on a large recording takes longer than the grace.
        if (
            editor.exit_on_disconnect
            and not editor.sockets
            and editor.on_shutdown is not None
            and editor.switching is None
        ):
            log.info("browser disconnected; stopping in %ss", editor.disconnect_grace)
            editor.pending_exit = asyncio.get_running_loop().call_later(
                editor.disconnect_grace, editor.on_shutdown
            )
