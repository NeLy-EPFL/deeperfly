"""The JSON and image payloads the editor's routes hand back.

Pure builders: each takes a :class:`~deeperfly.gui.session.Session` (or the arrays out
of one) and returns something JSON- or image-shaped. Nothing here touches the FastAPI
app, the socket set or the open-recording registry -- which is what makes this the half
of the server that can be tested by calling a function, and why the routes in
:mod:`deeperfly.gui.routes` stay thin enough to read as a list of endpoints.

The one piece of state is :class:`_EncodedFrames`, a byte-budgeted cache of already-JPEG
encoded frames: re-encoding dominates a frame request, and the same frame is asked for
by every panel of the page.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..labels.suggest import read_suggestions, suggestions_staleness
from ..visualization._palette import edge_colors_rgb, point_colors_rgb
from .session import Session

if TYPE_CHECKING:
    from ..skeleton import Skeleton

log = logging.getLogger("deeperfly")


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


def _render_mesh_png(session: Session, camera: str, t: int) -> bytes | None:
    """Render the posed model mesh for ``camera`` at frame ``t`` to RGBA PNG bytes.

    Sized to the camera's footage frame (so it overlays the served frame exactly).
    Returns ``None`` if the model or the packaged mesh asset is unavailable.
    """
    s = session.state
    if s.result.model_pts3d is None or camera not in s.result.cameras.names:
        return None
    try:
        from ..inverse_kinematics.mesh import load_model_mesh
        from ..visualization.mesh import render_mesh_rgba_auto
    except Exception:  # pragma: no cover -- a missing asset disables the overlay
        return None
    cam = s.result.cameras[camera]
    h, w = session.image_sizes.get(camera) or _intr_size(cam)
    mesh = load_model_mesh()
    angles = None if s.result.model_angles is None else s.result.model_angles[t]
    verts, valid = mesh.pose(
        s.result.model_pts3d[t],
        angles,
        s.result.model_angle_names,
        chain_scales=dict(s.result.model_chain_scales),
        chain_offsets=s.result.model_chain_offsets,
        body_scale=s.result.model_body_scale,
    )
    valid = np.asarray(valid) & ~mesh.hidden_face_mask(session.model_hide_parts)
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


@functools.lru_cache(maxsize=2)
def _model_asset_bytes() -> bytes | None:
    """The static mesh topology + per-vertex colors, packed once for the client.

    Layout (little-endian): ``uint32 n_verts``, ``uint32 n_faces``,
    ``uint32[n_faces * 3]`` triangle indices, ``uint8[n_verts * 3]`` vertex RGB.
    """
    try:
        from ..inverse_kinematics.mesh import load_model_mesh
    except Exception:  # pragma: no cover -- a missing asset disables the overlay
        return None
    mesh = load_model_mesh()
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


def _model_verts_bytes(session: Session, t: int) -> bytes | None:
    """The posed vertices + smooth normals + valid-face mask for ``t`` (re-fit from edits).

    Layout (little-endian): ``float32[n_verts * 3]`` world vertices (NaN -> 0), then
    ``float32[n_verts * 3]`` smooth per-vertex normals, then ``uint8[n_faces]`` --
    ``1`` where all three of a face's vertices were posed. The normals let the client
    smooth-shade the overlay (no faceting), and are computed here once per frame (the
    head/abdomen size is the IK data estimate, not an operator knob).
    """
    posed = session.state.model_posed_verts(t)
    if posed is None:
        return None
    from ..inverse_kinematics.mesh import load_model_mesh
    from ..visualization.mesh import vertex_normals

    verts, valid = posed
    mesh = load_model_mesh()
    faces = mesh.faces
    # Hide the configured body parts (default: wings) by dropping their faces.
    valid = np.asarray(valid) & ~mesh.hidden_face_mask(session.model_hide_parts)
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


def _color_legend(skel: Skeleton, colors: np.ndarray) -> list[dict]:
    """One ``{name, color}`` swatch per distinct colour, for the client's legend.

    Grouped by COLOUR rather than by any structure the skeleton declares, because it
    declares none: a skeleton is points, edges, symmetries and colours, so the only
    grouping there is to show is the one the operator authored in
    ``[skeleton.point_colors]``.
    For ``fly38`` that is the same ten swatches the per-limb legend used to draw.

    Each group is labelled by the **shared prefix** of its points' names, trimmed of a
    trailing separator (``lf_thorax_coxa`` + ``lf_pretarsus`` -> ``lf``), falling back to the
    first point's own name when they share nothing -- which is what a hand-written colour
    table that groups unrelated points deserves. ``colors`` is the per-point RGB already
    computed for the overlay, so the legend and the canvas cannot disagree.
    """
    groups: dict[str, list[int]] = {}
    for i, hexc in enumerate(skel.point_colors):
        groups.setdefault(hexc, []).append(i)
    out: list[dict] = []
    for hexc, members in groups.items():
        names = [skel.point_names[i] for i in members]
        label = os.path.commonprefix(names).rstrip("_-") if len(names) > 1 else names[0]
        out.append(
            {
                "name": label or names[0],
                "color": [int(c) for c in colors[members[0]]],
            }
        )
    return out


def _meta_payload(
    session: Session,
    cache_v: str | None = None,
    *,
    dirty_recordings: "Sequence[str]" = (),
) -> dict:
    """The one-time metadata the front-end needs to lay out and draw the editor.

    ``cache_v`` is the recording token the server will validate frame URLs against;
    it is passed in (rather than recomputed) so the two can never disagree.

    ``dirty_recordings`` names every recording the editor holds with unsaved labels --
    this one included when it is one of them. The front-end needs it on the very first
    payload of a page load or a switch: the beforeunload guard and the close prompt are
    project-wide, and a page that had only this recording's ``dirty`` would let an
    operator close the editor on another recording's unsaved work.
    """
    s = session.state
    skel = s.result.skeleton
    colors = (np.asarray(point_colors_rgb(skel)) * 255).round().astype(int)
    edge_colors = (np.asarray(edge_colors_rgb(skel)) * 255).round().astype(int)
    return {
        "results_path": session.results_path,
        # Stamped into the frame/mesh URLs so one recording's images can never be
        # served from cache for another -- see `_session_version`.
        "cache_v": cache_v if cache_v is not None else _session_version(session),
        "n_views": s.n_views,
        "n_frames": session.n_frames,
        "n_points": s.n_points,
        "has_3d": s.has_3d,
        "has_model": s.has_model,
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
        "camera_names": list(s.camera_names),
        "image_sizes": {
            name: [int(h), int(w)] for name, (h, w) in session.image_sizes.items()
        },
        "point_names": list(skel.point_names),
        # Printed in the title bar: `fly38@a1b2c3d4`, the label an error would quote.
        "skeleton_label": skel.label,
        "edges": np.asarray(skel.edges, dtype=int).reshape(-1, 2).tolist(),
        "point_colors": colors.tolist(),
        # Per EDGE, not looked up through an endpoint: an edge the skeleton colored
        # explicitly is not any of its points' color.
        "edge_colors": edge_colors.tolist(),
        "color_legend": _color_legend(skel, colors),
        "cameras_3d": _cameras_3d(session),
        "cameras_proj": _cameras_proj(session),
        "dirty": bool(s.dirty),
        "dirty_recordings": list(dirty_recordings),
        "project_dirty": bool(s.dirty or dirty_recordings),
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
    include_model: bool = True,
    verbose: bool = False,
) -> dict:
    """The per-view 2D overlay (with the fixed/invisible masks) for frame ``t`` in ``mode``.

    ``proj`` is the current 3D estimate reprojected into every view (with no fixed
    overrides) -- the display-only "latent skeleton" the front-end can ghost over
    every view; it is ``null`` when the result carries no 3D points.

    ``include_model`` (default ``True``) controls whether the fitted-model overlay is
    computed. The model reprojection needs a per-frame inverse-kinematics re-fit, so a
    live 3D drag -- which streams one edit per animation frame -- passes ``False`` to
    skip it (the ``model`` key is then *omitted*, and the front-end keeps the overlay it
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
    # Wire-compat mask names: "fixed" means "carries a GT pixel" and "invisible" is the
    # **hidden** flag -- "hold this cell out of the training loss". The two are independent,
    # so a cell may appear in both, and neither says anything about `points`: `invisible`
    # selects a mark drawn over the joint, never the joint's position.
    fixed = s.gt_mask(t)  # (V, P)
    invisible = s.occluded_mask(t)  # (V, P)
    # Absence is per-(frame, point), but this frame's row is broadcast to (V, P) so the
    # front-end indexes it exactly like `fixed` / `invisible`. It must ride the lean
    # mid-drag reply too: it gates whether a joint is drawn at all, so omitting it would
    # flash the phantom limb back on for the duration of every drag.
    absent = np.broadcast_to(s.absent_mask(t)[None, :], fixed.shape)
    # Cells whose drawn position the EDITOR invented: no evidence-backed seed and no
    # reprojection either, so `display_instance_pts2d` fell back to the placeholder chain (a
    # neighbour mean, the view centroid, the image centre). They look identical to a
    # triangulated joint on screen, `confirm` silently skips them, and the operator has no way
    # to tell -- so the marker has to say so. Rides every reply, like `fixed`: adding GT can
    # make a point solvable, which un-invents its cells.
    invented = s.invented_mask(t)
    proj = s.display_pts3d_projected(t) if s.has_3d else None
    payload = {
        "frame": t,
        "mode": mode,
        "points": _points_to_json(np.asarray(pts)),
        "instance": None if instance is None else _points_to_json(np.asarray(instance)),
        "has_instance": s.has_instance(t),
        "fixed": fixed.tolist(),
        "invisible": invisible.tolist(),
        "absent": np.asarray(absent).tolist(),
        "invented": invented.tolist(),
        # Which of those are absent in EVERY frame, so the front-end can tell an
        # amputation from a single-frame declaration without fetching the whole mask.
        "absent_recording": s.absent_points(),
        "proj": None if proj is None else _points_to_json(np.asarray(proj)),
        # One bool, on every reply including the lean mid-drag stream: the Reviewed control
        # sits on the frame row and must not lag behind the debounced corrected-frames refresh.
        "reviewed": bool(s.labels.reviewed[t]),
        "dirty": bool(s.dirty),
        "can_undo": s.can_undo,
        "can_redo": s.can_redo,
    }
    # The heavier per-point fields (detector confidence; raw predictions and the
    # missing-joint placeholder seeds for the verbose overlay) are static enough within
    # a frame to ride only the settle/plain reply -- not the ~60x/s mid-drag stream --
    # and the raw prediction / placeholder only when the verbose overlay is on.
    if include_model:
        proj = s.display_model_projected(t) if s.has_model else None
        payload["model"] = None if proj is None else _points_to_json(np.asarray(proj))
        payload["conf"] = _conf_to_json(s.result.conf, t)
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


def _conf_to_json(conf: np.ndarray | None, t: int) -> list | None:
    """``(V, P)`` detector confidence for frame ``t`` as nested lists (null if absent)."""
    if conf is None:
        return None
    c = np.asarray(conf[:, t], dtype=float)  # (V, P)
    return [
        [float(c[v, p]) if np.isfinite(c[v, p]) else None for p in range(c.shape[1])]
        for v in range(c.shape[0])
    ]


def _scene_payload(session: Session, t: int) -> dict:
    """The frame's 3D pose for the scene view: the triangulated keypoints and the
    fitted model joints (both world frame, skeleton order), each ``null`` when
    unavailable (2D-only, or no inverse-kinematics model)."""
    s = session.state
    pts3d = s.display_pts3d(t)
    fit = s.model_fit(t) if s.has_model else None
    model3d = None if fit is None else np.asarray(fit[0])
    return {
        "frame": t,
        "points3d": None if pts3d is None else _points3d_to_json(np.asarray(pts3d)),
        "model3d": None if model3d is None else _points3d_to_json(model3d),
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

    Thin wrapper over :func:`deeperfly.labels.suggest.read_suggestions`, which already
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
    :func:`deeperfly.labels.suggest.suggestions_staleness`, so they are decided in one place:
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


class _EncodedFrames:
    """A byte-budgeted LRU of encoded frame JPEGs, keyed ``(recording, camera, frame)``.

    The browser is the first line of defense and by far the best one: a frame URL stamped
    with this session's token is served ``immutable``, so scrubbing back to a frame *this
    tab* has already shown never reaches the server at all. This is the second line -- a
    reload, a second tab, the read-only viewers watching along -- each of which would
    otherwise pay the decode again for a picture already produced.

    Holding the *encoded* frame is what makes it affordable: ~250 KiB against 1.1 MiB for
    the same 1984x512 monochrome frame decoded (3.3 MiB in color), so a given budget covers
    roughly four times the frames the decoded cache behind it can.

    Keyed by the recording token for the same reason ``mesh_cache`` is: ``/api/frame/f/1506``
    names a different picture in every recording, and this cache sits inside the HTTP cache
    that token protects.

    The handlers run in a threadpool, so every access takes the lock -- an ``OrderedDict``
    survives concurrent readers but not a ``move_to_end`` racing an eviction.
    """

    def __init__(self, budget: int = 64 * 1024 * 1024) -> None:
        self._items: OrderedDict[tuple[str, str, int], bytes] = OrderedDict()
        self._budget = budget
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: tuple[str, str, int]) -> bytes | None:
        """The cached JPEG for ``key``, or ``None``; a hit becomes the most recent entry."""
        with self._lock:
            data = self._items.get(key)
            if data is not None:
                self._items.move_to_end(key)
            return data

    def put(self, key: tuple[str, str, int], data: bytes) -> None:
        """Cache ``data``, evicting the least recently used until inside the budget."""
        with self._lock:
            if key in self._items:
                self._bytes -= len(self._items.pop(key))
            self._items[key] = data
            self._bytes += len(data)
            while self._bytes > self._budget and self._items:
                self._bytes -= len(self._items.popitem(last=False)[1])


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """An ``(H, W, 3)`` RGB (or ``(H, W)`` gray) frame as the BGR cv2 expects.

    ``FrameSource`` yields RGB; ``cv2.imencode`` reads BGR, so the channels are
    reversed (a contiguous copy) before encoding to keep colors correct in the
    browser. Grayscale frames pass straight through.
    """
    if img.ndim == 2:
        return img
    return np.ascontiguousarray(img[..., ::-1])
