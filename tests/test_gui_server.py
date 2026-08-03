"""Tests for the web GUI server (:mod:`deeperfly.gui.server`), driven in-process.

FastAPI's ``TestClient`` exercises the HTTP API and the edit WebSocket against a
:class:`~deeperfly.gui.session.Session` built on the synthetic 7-camera fixture
with blank frames (no footage), so there is no real server, browser, or video
decoding involved. The edit ops themselves are covered in ``test_gui.py``; these
tests check the request/response wiring on top of them.
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient
from helpers import HEIGHT, WIDTH
from websockets.sync.client import connect as ws_connect

from deeperfly.gui import EditorState, FrameSource, Session, server
from deeperfly.gui.server import create_app


@pytest.fixture
def session(result, tmp_path):
    image_sizes = {name: (HEIGHT, WIDTH) for name in result.cameras.names}
    source = FrameSource({}, image_sizes=image_sizes)  # blank frames, no footage
    state = EditorState.from_result(result)
    return Session.build(
        state,
        source,
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
        image_sizes=image_sizes,
    )


@pytest.fixture
def client(session):
    return TestClient(create_app(session))


# -- metadata + frames --------------------------------------------------------


def test_meta_payload(client, result):
    meta = client.get("/api/meta").json()
    assert meta["n_views"] == result.n_views
    assert meta["n_frames"] == result.n_frames
    assert meta["n_points"] == result.pts2d.shape[2]
    assert meta["has_3d"] is True
    assert meta["has_nmf"] is False  # the default result carries no fitted NMF model
    assert list(meta["camera_names"]) == list(result.cameras.names)
    assert len(meta["point_colors"]) == result.pts2d.shape[2]
    assert len(meta["bones"]) == len(result.skeleton.bones)
    # The colour legend is data-driven from the skeleton: one {name, color} per limb,
    # so the front-end never hard-codes a left/right palette.
    limbs = meta["limbs"]
    assert [lb["name"] for lb in limbs] == list(result.skeleton.limb_names)
    assert all(len(lb["color"]) == 3 for lb in limbs)
    assert meta["dirty"] is False


def test_meta_cameras_3d(client, result):
    meta = client.get("/api/meta").json()
    cams = meta["cameras_3d"]
    assert [c["name"] for c in cams] == list(result.cameras.names)
    for name, cam in zip(result.cameras.names, result.cameras):
        entry = next(c for c in cams if c["name"] == name)
        assert np.allclose(entry["position"], cam.position, atol=1e-6)
        # forward is the camera's optical axis (third row of the rotation matrix)
        assert np.allclose(entry["forward"], cam.rmat[2], atol=1e-6)
        # the reported axes are unit length
        for axis in ("right", "up", "forward"):
            assert np.isclose(np.linalg.norm(entry[axis]), 1.0, atol=1e-6)


def test_frame_returns_jpeg(client, result):
    cam = result.cameras.names[0]
    r = client.get(f"/api/frame/{cam}/0")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"  # JPEG start-of-image marker


def test_frame_unknown_camera_404(client):
    assert client.get("/api/frame/nope/0").status_code == 404


def test_frames_are_only_cacheable_when_stamped_for_this_recording(client, result):
    """Two recordings served on the same port must not share cached frames.

    ``/api/frame/f/1506`` names a different picture in every recording, every
    ``deeperfly gui`` binds the same default port, and the suggestion queue picks
    near-identical frame indices across equal-length recordings -- so a long
    ``max-age`` on an unstamped URL made the browser paint the previously-opened
    recording's fly. Only a URL carrying this session's token may be cached.
    """
    cam = result.cameras.names[0]
    token = client.get("/api/meta").json()["cache_v"]
    assert token

    stamped = client.get(f"/api/frame/{cam}/0?v={token}")
    assert "immutable" in stamped.headers["cache-control"]

    # No stamp, or a stamp minted for a different recording: never cacheable.
    assert client.get(f"/api/frame/{cam}/0").headers["cache-control"] == "no-store"
    other = client.get(f"/api/frame/{cam}/0?v=deadbeefcafe")
    assert other.headers["cache-control"] == "no-store"


def test_cache_token_differs_between_recordings(session, tmp_path, result):
    """The token is derived from the recording, so two sessions never collide."""
    image_sizes = {name: (HEIGHT, WIDTH) for name in result.cameras.names}
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = Session.build(
        EditorState.from_result(result),
        FrameSource({}, image_sizes=image_sizes),
        results_path=str(other_dir / "results.h5"),
        labels_path=other_dir / "labels.h5",
        image_sizes=image_sizes,
    )
    assert server._session_version(session) != server._session_version(other)


def test_cache_token_survives_a_results_rewrite(result, tmp_path):
    """Labelling must not retire the token, or every frame is refetched for nothing.

    Declaring a keypoint absent mirrors the declaration into ``results.h5``. The
    served pixels come from the footage, not from that file, so the token is keyed on
    the footage and must be indifferent to the rewrite.
    """
    image_sizes = {name: (HEIGHT, WIDTH) for name in result.cameras.names}
    video = tmp_path / "camera_a.mp4"
    video.write_bytes(b"not a real video, only its identity is hashed")
    results_path = tmp_path / "results.h5"
    results_path.write_bytes(b"before")

    def build():
        source = FrameSource({}, image_sizes=image_sizes)
        source._files = {"a": [video]}  # resolved footage, without opening a decoder
        return Session.build(
            EditorState.from_result(result),
            source,
            results_path=str(results_path),
            labels_path=tmp_path / "labels.h5",
            image_sizes=image_sizes,
        )

    before = server._session_version(build())
    results_path.write_bytes(b"after the absence declaration was mirrored in")
    assert server._session_version(build()) == before

    # Re-pointing the result at different footage *does* retire it.
    video.write_bytes(b"a different recording's video file entirely")
    assert server._session_version(build()) != before


def test_index_declares_favicon(client):
    # The page head points the browser tab icon at the deeperfly logo. The .ico must
    # come before the SVG: Safari renders ICO but not SVG favicons and won't fall
    # back to a later <link>, so an SVG-first order leaves it drawing a letter tile.
    html = client.get("/").text
    assert 'rel="icon"' in html
    assert "/favicon.ico" in html
    assert "/static/logo.svg" in html
    assert html.index("/favicon.ico") < html.index("/static/logo.svg")
    # Safari draws favicons on a rounded chip; the apple-touch-icon fills it cleanly.
    assert 'rel="apple-touch-icon"' in html
    assert "/apple-touch-icon.png" in html


def test_favicon_served_from_root(client):
    # Safari probes /favicon.ico directly (not /static/...), so it must resolve there.
    ico = client.get("/favicon.ico")
    assert ico.status_code == 200
    assert ico.headers["content-type"] in ("image/x-icon", "image/vnd.microsoft.icon")
    assert ico.content[:4] == b"\x00\x00\x01\x00"  # ICO file header magic


def test_apple_touch_icon_served_from_root(client):
    # Safari probes /apple-touch-icon.png at the root for its Start Page / dock tiles.
    r = client.get("/apple-touch-icon.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature


def test_favicon_is_full_bleed_tile(client):
    # The .ico is a solid teal tile (the disc colour taken to the edges), not a bare
    # circle -- so it fills Safari's rounded chip instead of leaving a light box.
    # Every corner must be opaque; a circular icon would have transparent corners.
    from io import BytesIO

    from PIL import Image

    ico = Image.open(BytesIO(client.get("/favicon.ico").content))
    ico.size = (32, 32)
    px = ico.convert("RGBA").load()
    w, h = 32, 32
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        assert px[x, y][3] == 255, f"corner {(x, y)} should be opaque, got {px[x, y]}"


def test_favicon_svg_asset_is_served(client):
    svg = client.get("/static/logo.svg")
    assert svg.status_code == 200
    assert svg.headers["content-type"].startswith("image/svg+xml")
    assert svg.text.lstrip().startswith("<svg")


def test_placeholder_seeds_ride_the_verbose_reply(client, result):
    """The Missing-seed array is a verbose-only field, one seed per (view, point)."""
    plain = client.get("/api/points/0?mode=view").json()
    assert "placeholder" not in plain  # lean reply omits it
    verbose = client.get("/api/points/0?mode=view&verbose=true").json()
    ph = verbose["placeholder"]
    assert ph is not None and len(ph) == result.n_views
    assert all(len(row) == result.pts2d.shape[2] for row in ph)


def test_placeholder_seed_for_a_rejected_point(result, tmp_path):
    """A point NaN in every view (a triangulation reject) gets a finite draggable seed."""
    p = 5
    result.pts2d[:, :, p] = np.nan
    result.pts3d[:, p] = np.nan
    image_sizes = {name: (HEIGHT, WIDTH) for name in result.cameras.names}
    session = Session.build(
        EditorState.from_result(result, image_sizes=image_sizes),
        FrameSource({}, image_sizes=image_sizes),
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
        image_sizes=image_sizes,
    )
    client = TestClient(create_app(session))
    ph = client.get("/api/points/0?mode=view&verbose=true").json()["placeholder"]
    assert all(ph[v][p] is not None for v in range(result.n_views))  # every view seeded


def test_nmf_overlay_payload(result, tmp_path):
    """A result with a fitted NMF model exposes it via has_nmf + the points 'nmf' field."""
    import dataclasses

    # reuse the 3D points as a stand-in model; the server just reprojects them.
    res = dataclasses.replace(result, nmf_pts3d=np.asarray(result.pts3d))
    image_sizes = {name: (HEIGHT, WIDTH) for name in res.cameras.names}
    session = Session.build(
        EditorState.from_result(res),
        FrameSource({}, image_sizes=image_sizes),
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
        image_sizes=image_sizes,
    )
    client = TestClient(create_app(session))

    assert client.get("/api/meta").json()["has_nmf"] is True
    payload = client.get("/api/points/0?mode=view").json()
    nmf = payload["nmf"]
    assert nmf is not None and len(nmf) == res.n_views
    assert all(len(row) == res.pts2d.shape[2] for row in nmf)
    # the model reprojects to finite pixels in at least one view
    assert any(q is not None for row in nmf for q in row)

    # the posed-mesh overlay endpoint returns an RGBA PNG for a known camera
    cam = res.cameras.names[0]
    r = client.get(f"/api/mesh/{cam}/0")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:4] == b"\x89PNG"


def test_mesh_overlay_absent_without_ik(client):
    """The mesh overlay endpoint 404s when the result carries no fitted model."""
    assert client.get("/api/mesh/rf/0").status_code == 404


def _nmf_client(result, tmp_path):
    """A TestClient over a result carrying a (stand-in) fitted NMF model."""
    import dataclasses

    res = dataclasses.replace(result, nmf_pts3d=np.asarray(result.pts3d))
    image_sizes = {name: (HEIGHT, WIDTH) for name in res.cameras.names}
    session = Session.build(
        EditorState.from_result(res),
        FrameSource({}, image_sizes=image_sizes),
        results_path=str(tmp_path / "results.h5"),
        labels_path=tmp_path / "labels.h5",
        image_sizes=image_sizes,
    )
    return TestClient(create_app(session)), res


def test_nmf_client_render_asset_and_verts(result, tmp_path):
    """The client-render endpoints ship the mesh topology once + posed verts per frame."""
    from deeperfly.inverse_kinematics.mesh import load_nmf_mesh

    client, res = _nmf_client(result, tmp_path)
    mesh = load_nmf_mesh()
    n_v, n_f = int(mesh.vertices.shape[0]), int(mesh.faces.shape[0])

    asset = client.get("/api/nmf/asset")
    assert asset.status_code == 200
    head = np.frombuffer(asset.content[:8], dtype="<u4")
    assert head.tolist() == [n_v, n_f]
    assert len(asset.content) == 8 + n_f * 3 * 4 + n_v * 3  # header + faces + rgb

    verts = client.get("/api/nmf/verts/0")
    assert verts.status_code == 200
    # float32 verts + float32 smooth normals + uint8 valid mask
    assert len(verts.content) == n_v * 3 * 4 + n_v * 3 * 4 + n_f
    xyz = np.frombuffer(verts.content[: n_v * 12], dtype="<f4").reshape(n_v, 3)
    assert np.isfinite(xyz).all() and np.abs(xyz).sum() > 0  # the model is posed
    nrm = np.frombuffer(verts.content[n_v * 12 : n_v * 24], dtype="<f4").reshape(n_v, 3)
    lit = np.linalg.norm(nrm, axis=1) > 0.5  # posed faces carry a unit normal
    assert lit.any() and np.allclose(np.linalg.norm(nrm[lit], axis=1), 1.0, atol=1e-5)


def test_nmf_verts_payload_hides_configured_parts(result, tmp_path):
    """The verts payload drops the faces of the session's hidden parts (e.g. wings)."""
    import dataclasses

    from deeperfly.inverse_kinematics.mesh import load_nmf_mesh

    res = dataclasses.replace(result, nmf_pts3d=np.asarray(result.pts3d))
    image_sizes = {name: (HEIGHT, WIDTH) for name in res.cameras.names}
    mesh = load_nmf_mesh()
    n_v = int(mesh.vertices.shape[0])
    n_wing = int(mesh.hidden_face_mask(["wings"]).sum())
    assert n_wing > 0  # the baked asset labels wing faces

    def valid_sum(hide):
        session = Session.build(
            EditorState.from_result(res),
            FrameSource({}, image_sizes=image_sizes),
            results_path=str(tmp_path / "results.h5"),
            labels_path=tmp_path / "labels.h5",
            image_sizes=image_sizes,
            nmf_hide_parts=hide,
        )
        content = TestClient(create_app(session)).get("/api/nmf/verts/0").content
        return int(np.frombuffer(content[n_v * 24 :], dtype=np.uint8).sum())

    shown, hidden = valid_sum(()), valid_sum(("wings",))
    # the rigid body (which carries the wings) always poses, so hiding the wings
    # removes exactly their faces from the drawn (valid) set.
    assert shown - hidden == n_wing


def test_nmf_client_endpoints_absent_without_ik(client):
    """The client-render NMF endpoints 404 when the result carries no fitted model."""
    assert client.get("/api/nmf/asset").status_code == 404
    assert client.get("/api/nmf/verts/0").status_code == 404


def test_meta_cameras_proj_reproduces_projection(client, result):
    """meta.cameras_proj lets the client rebuild CameraGroup.project to sub-pixel.

    This is the contract the WebGL overlay relies on: the intrinsics + extrinsics +
    footage size in the payload must yield exactly the projection the server draws
    the other overlays with, or the mesh would not line up with the keypoints.
    """
    proj = client.get("/api/meta").json()["cameras_proj"]
    assert len(proj) == result.n_views
    pts3d = np.asarray(result.pts3d[0])
    pts3d = pts3d[np.isfinite(pts3d).all(axis=1)]
    near, far = 0.01, 1000.0
    flip = np.diag([1.0, -1.0, -1.0])
    for cam in proj:
        fx, fy, cx, cy = cam["intr"]
        w, h = cam["size"]
        rmat = np.asarray(cam["rmat"]).reshape(3, 3)
        view = np.eye(4)
        view[:3, :3] = flip @ rmat
        view[:3, 3] = flip @ np.asarray(cam["tvec"])
        pm = np.array(
            [
                [2 * fx / w, 0, (w - 2 * cx) / w, 0],
                [0, 2 * fy / h, (2 * cy - h) / h, 0],
                [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                [0, 0, -1, 0],
            ]
        )
        gl = []
        for x in pts3d:
            c = pm @ view @ np.array([*x, 1.0])
            n = c[:3] / c[3]
            gl.append([(n[0] * 0.5 + 0.5) * w, (0.5 - n[1] * 0.5) * h])
        ref = np.asarray(result.cameras[cam["name"]].project(pts3d))
        assert np.abs(np.asarray(gl) - ref).max() < 1e-2


# -- points -------------------------------------------------------------------


def test_points_payload_shapes(client, result):
    payload = client.get("/api/points/0?mode=view").json()
    assert payload["frame"] == 0
    n_points = result.pts2d.shape[2]
    assert len(payload["points"]) == result.n_views
    assert all(len(row) == n_points for row in payload["points"])
    assert len(payload["fixed"]) == result.n_views
    assert len(payload["invisible"]) == result.n_views
    assert all(len(row) == n_points for row in payload["invisible"])
    assert payload["dirty"] is False
    # every drawn point is either null or an [x, y] pair (NaN serializes to null)
    for row in payload["points"]:
        for pt in row:
            assert pt is None or len(pt) == 2


def test_points_payload_carries_latent_projection(client, result):
    # The latent overlay (the 3D estimate reprojected into every view) ships on
    # the points payload whenever the result has 3D, shaped like `points`.
    payload = client.get("/api/points/0?mode=edit_2d").json()
    proj = payload["proj"]
    assert proj is not None
    assert len(proj) == result.n_views
    assert all(len(row) == result.pts2d.shape[2] for row in proj)
    for row in proj:
        for pt in row:
            assert pt is None or len(pt) == 2


def test_scene_payload_has_3d_points(client, result):
    payload = client.get("/api/scene/0").json()
    assert payload["frame"] == 0
    pts3d = payload["points3d"]
    assert pts3d is not None
    assert len(pts3d) == result.pts2d.shape[2]
    for pt in pts3d:
        assert pt is None or len(pt) == 3
    assert payload["nmf3d"] is None  # the default result carries no fitted NMF model


def test_scene_payload_carries_nmf_skeleton(result, tmp_path):
    """When a model is fit, the 3D scene payload also ships its joints (for the viewer)."""
    client, res = _nmf_client(result, tmp_path)
    payload = client.get("/api/scene/0").json()
    nmf3d = payload["nmf3d"]
    assert nmf3d is not None
    assert len(nmf3d) == res.pts2d.shape[2]
    assert any(pt is not None and len(pt) == 3 for pt in nmf3d)


def test_nmf_live_uses_the_configured_model(result):
    """The editor's live re-fit honors a passed template, not always the full model.

    A run that restricts the fitted legs (or any IK config) must carry over to the
    editor, so the live overlay matches the pipeline fit rather than re-fitting every
    leg with the packaged default.
    """
    import dataclasses

    pytest.importorskip(
        "quickik", reason="the live re-fit needs the deeperfly[ik] extra"
    )
    from deeperfly.inverse_kinematics.template import KinematicTemplate

    res = dataclasses.replace(result, nmf_pts3d=np.asarray(result.pts3d))
    template = KinematicTemplate.load("neuromechfly", legs=["rf", "lf"])
    state = EditorState.from_result(res, template=template)
    assert state.nmf_live is not None
    leg_angles = [
        n for n in state.nmf_live.angle_names if "head" not in n and "abdomen" not in n
    ]
    # leg DOF names are "<parent>-<child>-<dof>"; the child body carries the leg code.
    assert {n.split("-")[1].split("_")[0] for n in leg_angles} == {"rf", "lf"}


def test_nmf_live_refit_is_a_pure_function_of_frame_and_labels(result):
    """The same frame, same labels, always the same angles -- however you got there.

    QuickIK's solver state is mutated in place and the editor solves arbitrary frames in
    arbitrary order as the operator scrubs, edits and undoes. A single long-lived state
    would make the answer depend on that history: the overlay would shift on undo/redo and
    nothing would be reproducible. So the seed depends only on the frame.
    """
    import dataclasses

    pytest.importorskip(
        "quickik", reason="the live re-fit needs the deeperfly[ik] extra"
    )

    res = dataclasses.replace(result, nmf_pts3d=np.asarray(result.pts3d))
    state = EditorState.from_result(res)
    assert state.nmf_live is not None
    live = state.nmf_live
    pts_a = np.asarray(res.pts3d)[0]
    pts_b = np.asarray(res.pts3d)[min(1, res.pts3d.shape[0] - 1)]

    first, angles_first = live.refit(pts_a, 0)
    live.refit(pts_b, 1)  # visit another frame in between
    again, angles_again = live.refit(pts_a, 0)
    np.testing.assert_array_equal(angles_first, angles_again)
    np.testing.assert_array_equal(first, again)


def test_nmf_live_masks_a_limb_it_cannot_fit(result):
    """A limb with too few observed keypoints comes back NaN, as in the batch fit.

    Otherwise the editor would draw a limb at whatever pose the neutral prior implied,
    with nothing to say it was never observed.
    """
    import dataclasses

    pytest.importorskip(
        "quickik", reason="the live re-fit needs the deeperfly[ik] extra"
    )

    res = dataclasses.replace(result, nmf_pts3d=np.asarray(result.pts3d))
    state = EditorState.from_result(res)
    live = state.nmf_live
    assert live is not None
    index = {n: i for i, n in enumerate(res.skeleton.point_names)}
    pts = np.asarray(res.pts3d)[0].copy()
    for name in ("rh_coxa_trochanter", "rh_femur_tibia", "rh_tibia_tarsus", "rh_claw"):
        pts[index[name]] = np.nan
    model, angles = live.refit(pts, 0)
    rh = [
        i
        for i, n in enumerate(live.angle_names)
        if "-rh_" in n or "rh_" in n.split("-")[0]
    ]
    assert rh, "the plan should carry right-hind leg DOFs"
    assert np.isnan(angles[rh]).all()
    assert np.isnan(model[index["rh_claw"]]).all()


def test_corrected_endpoint_lists_edited_frames(client):
    """/api/corrected reports the frames the operator touched, each with a reviewed flag."""
    assert client.get("/api/corrected").json()["frames"] == []  # nothing edited yet

    with client.websocket_connect("/ws") as ws:
        for view, point in ((0, 3), (1, 5)):  # two distinct points in frame 2
            ws.send_json(
                {
                    "type": "edit_2d",
                    "view": view,
                    "point": point,
                    "x": 12.0,
                    "y": 34.0,
                    "frame": 2,
                    "mode": "edit_2d",
                }
            )
            ws.receive_json()

    frames = client.get("/api/corrected").json()["frames"]
    assert [f["frame"] for f in frames] == [2]
    assert not frames[0]["reviewed"]  # edited, but not yet reviewed


def test_set_reviewed_edit_marks_frame(client):
    """A set_reviewed edit ticks a frame reviewed; it then shows in /api/corrected."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {"type": "set_reviewed", "reviewed": True, "frame": 3, "mode": "edit_2d"}
        )
        ws.receive_json()

    frames = client.get("/api/corrected").json()["frames"]
    assert [f["frame"] for f in frames] == [3]
    assert frames[0]["reviewed"]  # listed purely because it was marked reviewed

    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {"type": "set_reviewed", "reviewed": False, "frame": 3, "mode": "edit_2d"}
        )
        ws.receive_json()
    assert client.get("/api/corrected").json()["frames"] == []  # un-ticked -> dropped


# -- suggested frames (the labels-suggest queue) -------------------------------
#
# `/api/suggestions` is a READER of the `labels_suggest.json` sidecar: the ranking
# triangulates the whole recording, so it is a CLI artefact and the route only serves it,
# joined with the live labeled/reviewed state. These tests pin the three things a wrong
# panel would hide: that the queue is never presented as fresher than it is, that a frame
# flips to done from live state rather than from the file, and that every degenerate
# sidecar (absent, corrupt, future-format, out-of-range) degrades to an empty panel
# instead of an error or a plausible-looking lie.


def _suggestions_doc(frames, **extra) -> dict:
    """A minimal v1 sidecar: only the fields the route actually reads."""
    doc = {
        "deeperfly_suggestions_format_version": 1,
        "created_utc": "2026-07-29T08:40:11+00:00",
        "params": {"count": 20, "min_gap_s": 2.0, "threshold_px": 15.0},
        "source": {
            "scored_array": "pose2d/points",
            "cameras_from": "bundle_adjustment",
        },
        "coverage": {"global_residual_median_px": 6.35},
        "frames": frames,
    }
    doc.update(extra)
    return doc


def _write_suggestions(session, frames, **extra) -> None:
    session.suggestions_path.write_text(
        json.dumps(_suggestions_doc(frames, **extra)), encoding="utf-8"
    )


def _pick(frame, rank=1, **extra) -> dict:
    entry = {
        "rank": rank,
        "frame": frame,
        "t_s": frame / 100.0,
        "score": 0.6,
        "percentile": 99.0,
        "kind": "most-wrong",
        "reason": {"summary": f"4 views disagree at frame {frame}"},
    }
    entry.update(extra)
    return entry


def test_suggestions_absent_reports_how_to_make_one(client, session):
    """No sidecar is a normal state: `present: false` plus the exact command to run."""
    assert not session.suggestions_path.exists()
    payload = client.get("/api/suggestions").json()
    assert payload["present"] is False
    # The command names the resolved results directory, so the panel needs no doc lookup.
    assert payload["command"].startswith("deeperfly labels-suggest ")
    assert str(session.suggestions_path.parent) in payload["command"]


def test_suggestions_payload_carries_the_ranking_and_its_reason(client, session):
    """The queue is served in rank order with the per-pick "why" intact."""
    _write_suggestions(
        session,
        [
            _pick(3, rank=2, kind="diversity", reason={"summary": "grid slot 2/5"}),
            _pick(1, rank=1),
        ],
        shortfall={"requested": 20, "selected": 2, "reason": "spacing ran out of room"},
    )
    payload = client.get("/api/suggestions").json()
    assert payload["present"] is True
    assert [f["rank"] for f in payload["frames"]] == [
        1,
        2,
    ]  # sorted by rank, not by file
    assert [f["frame"] for f in payload["frames"]] == [1, 3]
    assert payload["frames"][1]["kind"] == "diversity"
    assert payload["frames"][0]["reason"]["summary"].startswith("4 views disagree")
    # Provenance rides along so the panel can say what was scored.
    assert payload["source"]["scored_array"] == "pose2d/points"
    assert payload["stale"]["level"] == "none"
    assert payload["n_done"] == 0
    assert all(not f["labeled"] for f in payload["frames"])
    # An under-delivered count is stated, not swallowed: 2 of 20 is the whole point.
    assert any("2 of 20 requested" in n for n in payload["notes"])


def test_suggestions_mark_labeled_frames_done(client, session):
    """A suggested frame the operator has since labeled comes back `labeled` + progress.

    The flag is taken from the same live state the Labels list uses, never from the
    sidecar -- which was written before this session's edits.
    """
    _write_suggestions(session, [_pick(2, rank=1), _pick(5, rank=2)])
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "edit_2d",
                "view": 0,
                "point": 3,
                "x": 12.0,
                "y": 34.0,
                "frame": 2,
                "mode": "edit_2d",
            }
        )
        ws.receive_json()

    payload = client.get("/api/suggestions").json()
    by_frame = {f["frame"]: f for f in payload["frames"]}
    assert by_frame[2]["labeled"] is True and by_frame[2]["reviewed"] is False
    assert by_frame[5]["labeled"] is False
    assert payload["n_done"] == 1
    # Working through the queue is progress, not a problem -- and it is reported as such.
    # The tier carries no reason string on purpose: the panel counts "k of n done" live
    # from the edits it has just made, so a sentence composed here would go stale.
    assert payload["stale"]["level"] == "progress"
    assert payload["stale"]["reasons"] == []


def test_suggestions_carry_the_reviewed_tick(client, session):
    """Ticking a frame reviewed shows on its queue entry (same source as the other list)."""
    _write_suggestions(session, [_pick(3, rank=1)])
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {"type": "set_reviewed", "reviewed": True, "frame": 3, "mode": "edit_2d"}
        )
        ws.receive_json()
    entry = client.get("/api/suggestions").json()["frames"][0]
    assert entry["labeled"] is True and entry["reviewed"] is True


def test_suggestions_for_another_recording_are_hard_stale(client, session):
    """A queue fingerprinting a different recording is refused, not silently navigated."""
    other = dict(session.identity)
    other["point_names"] = ["nose", "tail"]  # a different keypoint set entirely
    _write_suggestions(
        session,
        [_pick(1, rank=1)],
        source={"scored_array": "pose2d/points", "identity": other},
    )
    payload = client.get("/api/suggestions").json()
    assert payload["present"] is True
    assert payload["stale"]["level"] == "hard"
    assert "different recording" in payload["stale"]["reasons"][0]


def test_suggestions_matching_identity_is_not_stale(client, session):
    """The identity check must not fire on the recording the queue was computed for."""
    _write_suggestions(
        session,
        [_pick(1, rank=1)],
        source={"scored_array": "pose2d/points", "identity": session.identity},
    )
    assert client.get("/api/suggestions").json()["stale"]["level"] == "none"


def _fingerprint(path: Path) -> dict:
    """The results.h5 fingerprint a real sidecar records: stat first, then the hash."""
    stat = path.stat()
    return {
        "results_md5": hashlib.md5(path.read_bytes()).hexdigest(),
        "results_size": stat.st_size,
        "results_mtime_ns": stat.st_mtime_ns,
    }


def test_suggestions_detect_superseded_predictions(client, session):
    """A results.h5 rewritten since the queue was computed is flagged, by md5."""
    results = Path(session.results_path)
    results.write_bytes(b"the predictions the queue was computed from")
    stale_fingerprint = _fingerprint(results)
    results.write_bytes(b"re-run predictions, different content entirely")

    _write_suggestions(
        session,
        [_pick(1, rank=1)],
        source={"scored_array": "pose2d/points", **stale_fingerprint},
    )
    payload = client.get("/api/suggestions").json()
    assert payload["stale"]["level"] == "predictions"
    assert any("superseded" in r for r in payload["stale"]["reasons"])
    # The frames are real, so they still render -- the warning rides above them.
    assert [f["frame"] for f in payload["frames"]] == [1]

    # The same file, correctly fingerprinted, is not stale.
    _write_suggestions(
        session,
        [_pick(1, rank=1)],
        source={"scored_array": "pose2d/points", **_fingerprint(results)},
    )
    assert client.get("/api/suggestions").json()["stale"]["level"] == "none"


def test_suggestions_without_a_fingerprint_do_not_cry_wolf(client, session):
    """Not knowing whether results.h5 changed is not the same as knowing it did."""
    Path(session.results_path).write_bytes(b"pretend results")
    _write_suggestions(session, [_pick(1, rank=1)])  # no results_md5 recorded
    assert client.get("/api/suggestions").json()["stale"]["level"] == "none"


def test_suggestions_drop_frames_outside_the_recording(client, session):
    """A row that cannot be navigated to is dropped -- and the drop is reported."""
    beyond = session.n_frames + 500
    _write_suggestions(
        session, [_pick(1, rank=1), _pick(beyond, rank=2), _pick(-3, rank=3)]
    )
    payload = client.get("/api/suggestions").json()
    assert [f["frame"] for f in payload["frames"]] == [1]
    assert any("outside this recording" in n for n in payload["notes"])


def test_suggestions_note_a_reseeded_result(client, session):
    """A reseeded result says so: its STORED reprojection error would have ranked nothing."""
    _write_suggestions(
        session,
        [_pick(1, rank=1)],
        source={"scored_array": "pose2d/points", "reseeded": True},
    )
    notes = client.get("/api/suggestions").json()["notes"]
    assert any("reseeded" in n and "pose2d/points" in n for n in notes)


def test_suggestions_note_uncalibrated_cameras(client, session):
    """Scoring against the config rig may rank calibration error, so the panel warns."""
    _write_suggestions(
        session,
        [_pick(1, rank=1)],
        source={"scored_array": "pose2d/points", "cameras_from": "pose2d"},
        coverage={"global_residual_median_px": 40.0},
    )
    notes = client.get("/api/suggestions").json()["notes"]
    assert any("bundle adjustment" in n for n in notes)
    # A recording whose median disagreement already exceeds the threshold is the
    # signature of ranking calibration rather than the detector's mistakes.
    assert any("median disagreement" in n for n in notes)


@pytest.mark.parametrize(
    "text",
    [
        "{not json at all",
        json.dumps({"deeperfly_suggestions_format_version": 99, "frames": []}),
        json.dumps([1, 2, 3]),
    ],
    ids=["corrupt", "future-format", "not-an-object"],
)
def test_unusable_suggestions_sidecar_degrades_to_absent(client, session, text):
    """Anything unreadable or unrecognised reads as "no queue", never as an error."""
    session.suggestions_path.write_text(text, encoding="utf-8")
    payload = client.get("/api/suggestions").json()
    assert payload["present"] is False
    assert payload["command"].startswith("deeperfly labels-suggest ")


def test_malformed_suggestion_rows_are_skipped(client, session):
    """One bad row must not cost the whole queue."""
    _write_suggestions(session, [{"rank": 1}, "nonsense", _pick(4, rank=2)])
    payload = client.get("/api/suggestions").json()
    assert [f["frame"] for f in payload["frames"]] == [4]


def test_unformattable_numbers_become_null(client, session):
    """A bad numeric field reads as null, not as a string the front-end would choke on.

    The panel formats score / time / percentile with `toFixed`, which throws on a
    string -- so one hand-edited field would otherwise take the whole list down.
    """
    _write_suggestions(
        session, [_pick(1, rank=1, score="high", t_s=None, percentile="p99")]
    )
    entry = client.get("/api/suggestions").json()["frames"][0]
    assert entry["score"] is None and entry["t_s"] is None
    assert entry["percentile"] is None


def test_suggestions_never_write_anything(client, session):
    """The route is a reader: it must not touch the sidecar, the labels, or results.h5.

    `labels.h5` holds thousands of hand-placed points and each recording's `results.h5`
    holds the only copy of its calibration, so "reads only" is a property worth pinning.
    """
    results = Path(session.results_path)
    results.write_bytes(b"pretend results")
    session.labels_path.write_bytes(b"pretend labels")
    _write_suggestions(session, [_pick(1, rank=1)])
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in (results, session.labels_path, session.suggestions_path)
    }
    assert client.get("/api/suggestions").json()["present"] is True
    for path, (content, mtime) in before.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime_ns == mtime


def test_suggestions_go_through_the_acquisition_reader(client, session, monkeypatch):
    """The sidecar format is owned by `deeperfly.acquisition`, so ITS reader parses it.

    Pinning the delegation matters because the alternative is tempting and wrong: a
    second JSON parse in the GUI would be a second interpretation of the format, free to
    drift from the writer's (over the version gate especially) while still looking fine.
    """
    calls = []

    def read_suggestions(path):
        calls.append(Path(path))
        return _suggestions_doc([_pick(4, rank=1)])

    monkeypatch.setattr(server, "read_suggestions", read_suggestions)
    _write_suggestions(session, [_pick(1, rank=1)])  # different content than the stub
    payload = client.get("/api/suggestions").json()
    assert calls == [session.suggestions_path]
    assert [f["frame"] for f in payload["frames"]] == [4]


def test_suggestions_survive_a_broken_reader(client, session, monkeypatch):
    """A reader that raises leaves an empty panel, not a broken editor."""

    def read_suggestions(path):
        raise ValueError("boom")

    monkeypatch.setattr(server, "read_suggestions", read_suggestions)
    _write_suggestions(session, [_pick(1, rank=1)])
    assert client.get("/api/suggestions").json()["present"] is False


def test_session_defaults_the_suggestions_path_beside_the_labels(session):
    """The sidecar lives next to labels.h5 under the shared `labels_*` naming."""
    assert (
        session.suggestions_path == session.labels_path.parent / "labels_suggest.json"
    )


# -- the front-end wiring for the panel ---------------------------------------
#
# The Python tests cannot run the browser, but they can pin the contract between the
# three hand-edited assets: an id the HTML no longer has (or never had) makes `el(...)`
# return null and the whole editor dies on load -- the exact failure the ?v= asset stamp
# exists to prevent. Cheap insurance for a purely additive DOM change.


def test_suggest_panel_ids_agree_across_the_assets(client):
    html = client.get("/").text
    js = client.get("/static/app.js").text
    css = client.get("/static/styles.css").text
    for element_id in (
        "sidebar-tabs",
        "suggest-count",
        "labeled-pane",
        "suggest-pane",
        "suggest-status",
        "suggest-table",
        "suggest-empty",
    ):
        assert f'id="{element_id}"' in html, f"{element_id} missing from index.html"
        assert f'el("{element_id}")' in js, f"{element_id} not bound in app.js"
    # The two lists share the table styling; the queue adds its own row states + chips.
    for cls in ("suggest-table", "kind-chip", "sidebar-status", "suggest-why"):
        assert f".{cls}" in css, f"{cls} unstyled"
    # The queue is fetched from the route these tests cover, and the panel is a reader.
    assert "fetchSuggestions" in js
    assert "/api/suggestions" in client.get("/static/api.js").text


# -- edits over the websocket -------------------------------------------------


def test_ws_edit_3d_updates_all_views_and_sets_dirty(client, result):
    view, point = 2, 5
    base = client.get("/api/points/0?mode=edit_3d").json()["points"]
    target = [base[view][point][0] + 10.0, base[view][point][1] - 8.0]

    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "edit_3d",
                "view": view,
                "point": point,
                "x": target[0],
                "y": target[1],
                "frame": 0,
                "fix": False,
                "mode": "edit_3d",
            }
        )
        reply = ws.receive_json()

    assert reply["dirty"] is True
    # the dragged view's reprojection lands on the cursor
    assert np.allclose(reply["points"][view][point], target, atol=1e-3)
    # at least one other view's reprojection moved as the 3D point was re-solved
    other = (view + 1) % result.n_views
    assert not np.allclose(reply["points"][other][point], base[other][point])


def test_ws_toggle_invisible_flips_mask_and_sets_dirty(client, result):
    view, point = 1, 5
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "toggle_invisible",
                "view": view,
                "point": point,
                "frame": 0,
                "mode": "edit_3d",
            }
        )
        reply = ws.receive_json()
    assert reply["dirty"] is True
    assert reply["invisible"][view][point] is True
    # the marked view drops out of the estimate; the other views stay finite
    other = (view + 1) % result.n_views
    assert reply["invisible"][other][point] is False


def test_ws_occluded_cell_has_no_observation_but_keeps_its_reprojection(client, result):
    """An occluded ("Projected") cell must be null in ``points`` yet finite in ``proj``.

    That pair is the contract the editor's display precedence rests on: the view has no
    usable observation (so the rejected detection is not drawn there and does not feed
    the solve), and the reprojection of the re-solved 3D is what the joint falls back
    to as its position in that view.
    """
    view, point = 1, 5
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "occlude", "targets": [[view, point]], "frame": 0})
        reply = ws.receive_json()
    assert reply["invisible"][view][point] is True
    assert reply["points"][view][point] is None  # no observation left in this view
    assert reply["proj"][view][point] is not None  # ... but a reprojection to draw
    other = (view + 1) % result.n_views
    assert reply["points"][other][point] is not None  # other views keep theirs


def test_ws_edit_2d_is_local_to_its_view(client):
    view, point = 0, 3
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "edit_2d",
                "view": view,
                "point": point,
                "x": 12.0,
                "y": 34.0,
                "frame": 1,
                "mode": "edit_2d",
            }
        )
        reply = ws.receive_json()
    assert reply["frame"] == 1
    assert np.allclose(reply["points"][view][point], [12.0, 34.0])
    assert reply["dirty"] is True


def test_ws_reply_echoes_edit_seq(client):
    """Every edit reply echoes the sending edit's seq, so the front-end can drop a
    superseded reply (a mid-drag re-solve landing after release)."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "edit_2d",
                "view": 0,
                "point": 3,
                "x": 5.0,
                "y": 6.0,
                "frame": 0,
                "mode": "edit_2d",
                "seq": 42,
            }
        )
        reply = ws.receive_json()
    assert reply["seq"] == 42


def test_ws_live_drag_omits_nmf_then_includes_on_release(result, tmp_path):
    """A mid-drag edit_3d (fix=False) omits the 'nmf' key -- skipping the per-frame
    IK re-fit that dominated drag latency -- while the pin/release reply (fix=True)
    carries it again. The overlay's absence tells the client to hold what it has."""
    client, _res = _nmf_client(result, tmp_path)
    view, point = 2, 5
    base = client.get("/api/points/0?mode=edit_3d").json()["points"]
    target = [base[view][point][0] + 3.0, base[view][point][1] - 2.0]

    with client.websocket_connect("/ws") as ws:
        drag = {
            "type": "edit_3d",
            "view": view,
            "point": point,
            "x": target[0],
            "y": target[1],
            "frame": 0,
            "mode": "edit_3d",
        }
        ws.send_json({**drag, "fix": False, "seq": 1})
        mid = ws.receive_json()
        ws.send_json({**drag, "fix": True, "seq": 2})
        end = ws.receive_json()

    assert "nmf" not in mid and mid["seq"] == 1  # mid-drag: refit skipped
    assert end["nmf"] is not None and end["seq"] == 2  # release: overlay recomputed


def test_ws_reset_point_view_reverts_one_view(client):
    point = 3
    with client.websocket_connect("/ws") as ws:
        for view, xy in ((0, (12.0, 34.0)), (1, (56.0, 78.0))):
            ws.send_json(
                {
                    "type": "edit_2d",
                    "view": view,
                    "point": point,
                    "x": xy[0],
                    "y": xy[1],
                    "frame": 0,
                    "mode": "edit_2d",
                }
            )
            ws.receive_json()
        ws.send_json(
            {
                "type": "reset_point_view",
                "view": 0,
                "point": point,
                "frame": 0,
                "mode": "edit_2d",
            }
        )
        reply = ws.receive_json()
    # View 0 reverts off its edit; view 1's edit is untouched.
    assert not np.allclose(reply["points"][0][point], [12.0, 34.0])
    assert np.allclose(reply["points"][1][point], [56.0, 78.0])


def test_ws_reset_frame_reverts_every_point(client):
    with client.websocket_connect("/ws") as ws:
        for view, point, xy in ((0, 3, (12.0, 34.0)), (1, 5, (56.0, 78.0))):
            ws.send_json(
                {
                    "type": "edit_2d",
                    "view": view,
                    "point": point,
                    "x": xy[0],
                    "y": xy[1],
                    "frame": 0,
                    "mode": "edit_2d",
                }
            )
            ws.receive_json()
        ws.send_json({"type": "reset_frame", "frame": 0, "mode": "edit_2d"})
        reply = ws.receive_json()
    # Both edits are gone from the frame.
    assert not np.allclose(reply["points"][0][3], [12.0, 34.0])
    assert not np.allclose(reply["points"][1][5], [56.0, 78.0])


def test_ws_confirm_promotes_predictions(client, result):
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "confirm",
                "targets": [[v, 5] for v in range(result.n_views)],
                "sources": "predictions",
                "frame": 0,
                "mode": "edit_3d",
                "seq": 1,
            }
        )
        reply = ws.receive_json()
    assert reply["dirty"] is True
    assert all(reply["fixed"][v][5] for v in range(result.n_views))  # all GT now


def test_ws_reset_targets_reverts_selected_cells_in_one_step(client):
    # The batched "Reset" acting on a multi-cell selection: it reverts exactly the
    # targeted cells (leaving the rest) and is a single undoable step.
    with client.websocket_connect("/ws") as ws:
        for view, point in ((0, 3), (1, 3), (0, 5)):
            ws.send_json(
                {
                    "type": "edit_2d",
                    "view": view,
                    "point": point,
                    "x": 12.0,
                    "y": 34.0,
                    "frame": 0,
                    "mode": "edit_2d",
                }
            )
            ws.receive_json()
        ws.send_json(
            {
                "type": "reset",
                "targets": [[0, 3], [1, 3]],
                "frame": 0,
                "mode": "edit_2d",
                "seq": 9,
            }
        )
        reply = ws.receive_json()
        assert reply["fixed"][0][3] is False  # targeted -> reverted
        assert reply["fixed"][1][3] is False
        assert reply["fixed"][0][5] is True  # untargeted -> untouched
        assert reply["can_undo"] is True
        # A single undo restores both reset cells (one step for the whole batch).
        ws.send_json({"type": "undo", "frame": 0, "mode": "edit_2d", "seq": 10})
        undo = ws.receive_json()
        assert undo["fixed"][0][3] is True
        assert undo["fixed"][1][3] is True


def test_ws_reset_targets_empty_is_a_noop(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {"type": "reset", "targets": [], "frame": 0, "mode": "edit_2d", "seq": 1}
        )
        reply = ws.receive_json()
    assert reply["dirty"] is False
    assert reply["can_undo"] is False


def test_ws_occlude_targets_marks_cells_in_one_step(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "occlude",
                "targets": [[0, 4], [1, 4]],
                "frame": 0,
                "mode": "edit_3d",
                "seq": 1,
            }
        )
        reply = ws.receive_json()
        assert reply["invisible"][0][4] is True
        assert reply["invisible"][1][4] is True
        assert reply["dirty"] is True
        assert reply["can_undo"] is True
        # One undo clears both occlusions.
        ws.send_json({"type": "undo", "frame": 0, "mode": "edit_3d", "seq": 2})
        undo = ws.receive_json()
        assert undo["invisible"][0][4] is False
        assert undo["invisible"][1][4] is False


def test_ws_occlude_targets_keeps_any_gt(client):
    # Occlusion is orthogonal to GT now: marking a labeled cell not-visible records both.
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "edit_2d",
                "view": 0,
                "point": 2,
                "x": 10.0,
                "y": 20.0,
                "frame": 0,
                "mode": "edit_2d",
            }
        )
        ws.receive_json()
        ws.send_json(
            {
                "type": "occlude",
                "targets": [[0, 2]],
                "frame": 0,
                "mode": "edit_3d",
                "seq": 5,
            }
        )
        reply = ws.receive_json()
    assert reply["fixed"][0][2] is True  # the pixel stands
    assert reply["invisible"][0][2] is True  # ... and so does "not visible here"


def test_ws_reset_targets_noop_stays_clean(client):
    # A Reset over cells that carry no label (e.g. select-all then Reset on a fresh
    # frame) changes nothing: no dirty flip, no undo entry.
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "reset",
                "targets": [[0, 1], [1, 1]],
                "frame": 0,
                "mode": "edit_2d",
            }
        )
        reply = ws.receive_json()
    assert reply["dirty"] is False
    assert reply["can_undo"] is False


def test_ws_occlude_targets_noop_pushes_no_undo(client):
    # Occluding an already-occluded cell is a no-op: it must not push a phantom undo
    # step, so a single undo fully reverts the (one real) occlusion.
    occ = {"type": "occlude", "targets": [[0, 3]], "frame": 0, "mode": "edit_3d"}
    with client.websocket_connect("/ws") as ws:
        ws.send_json(occ)
        assert ws.receive_json()["invisible"][0][3] is True
        ws.send_json(occ)
        assert ws.receive_json()["invisible"][0][3] is True  # still occluded (no-op)
        ws.send_json({"type": "undo", "frame": 0, "mode": "edit_3d"})
        assert ws.receive_json()["invisible"][0][3] is False  # one undo clears it


def test_ws_confirm_noop_preserves_redo(client):
    # A no-op Confirm (targeting an already-GT cell) must NOT clobber the redo stack.
    with client.websocket_connect("/ws") as ws:
        for point in (1, 2):
            ws.send_json(
                {
                    "type": "edit_2d",
                    "view": 0,
                    "point": point,
                    "x": 5.0,
                    "y": 6.0,
                    "frame": 0,
                    "mode": "edit_2d",
                }
            )
            ws.receive_json()
        ws.send_json({"type": "undo", "frame": 0, "mode": "edit_2d"})
        assert ws.receive_json()["can_redo"] is True  # point 2's edit is redoable
        # Confirm a cell that is already GT -> nothing changes; redo must survive.
        ws.send_json(
            {
                "type": "confirm",
                "targets": [[0, 1]],
                "sources": "all",
                "frame": 0,
                "mode": "edit_2d",
            }
        )
        assert ws.receive_json()["can_redo"] is True


def test_ws_undo_redo_carry_the_target_frame(client, result):
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "edit_2d",
                "view": 0,
                "point": 3,
                "x": 12.0,
                "y": 34.0,
                "frame": 1,
                "mode": "edit_2d",
                "seq": 1,
            }
        )
        ws.receive_json()
        ws.send_json({"type": "undo", "frame": 0, "mode": "edit_2d", "seq": 2})
        undo = ws.receive_json()
        assert undo["goto"] == 1  # the edit was on frame 1
        assert undo["fixed"][0][3] is False  # GT reverted
        ws.send_json({"type": "redo", "frame": 0, "mode": "edit_2d", "seq": 3})
        redo = ws.receive_json()
        assert redo["goto"] == 1
        assert redo["fixed"][0][3] is True  # GT re-applied


def test_points_payload_carries_conf_and_undo_flags(client):
    pay = client.get("/api/points/0?mode=view").json()
    assert pay["conf"] is not None and len(pay["conf"]) > 0
    assert pay["can_undo"] is False and pay["can_redo"] is False
    verbose = client.get("/api/points/0?mode=view&verbose=true").json()
    assert "pred" in verbose and verbose["pred"] is not None


def test_save_writes_sidecar_and_clears_dirty(client, session):
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "edit_2d",
                "view": 0,
                "point": 1,
                "x": 5.0,
                "y": 6.0,
                "frame": 0,
                "mode": "edit_2d",
            }
        )
        ws.receive_json()
    assert session.state.dirty

    resp = client.post("/api/save").json()
    assert resp["dirty"] is False
    assert session.labels_path.exists()
    assert not session.state.dirty


# -- shutdown -----------------------------------------------------------------


def test_shutdown_invokes_hook(session):
    # The Close button POSTs /api/shutdown; the server calls the wired-in hook
    # (serve() flips uvicorn's should_exit) and replies before stopping.
    calls = []
    client = TestClient(create_app(session, on_shutdown=lambda: calls.append(True)))
    resp = client.post("/api/shutdown")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert calls == [True]


def test_shutdown_without_hook_is_a_noop(client):
    # No hook wired in (e.g. in tests): the endpoint still answers cleanly.
    assert client.post("/api/shutdown").status_code == 200


# -- auto-shutdown when the browser disconnects -------------------------------
#
# These run a real uvicorn server in a daemon thread (the in-process TestClient
# tears its event loop down with each websocket, so it can't exercise the
# grace-period timer) and drive it with a real WebSocket client.


def _serve(session, **create_kw):
    """Run ``create_app(session, **create_kw)`` under uvicorn on a free port.

    Returns ``(server, thread, port)``. ``on_shutdown`` flips the server's
    ``should_exit`` exactly as :func:`deeperfly.gui.serve` wires it.
    """
    holder: dict = {}
    app = create_app(
        session,
        on_shutdown=lambda: setattr(holder["server"], "should_exit", True),
        **create_kw,
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    holder["server"] = server
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "server did not start"
    return server, thread, port


def test_closing_the_last_tab_stops_the_server(session):
    server, thread, port = _serve(
        session, exit_on_disconnect=True, disconnect_grace=0.3
    )
    with ws_connect(f"ws://127.0.0.1:{port}/ws"):
        pass  # a browser tab opens its socket, then closes it (the tab is closed)
    thread.join(timeout=5)
    assert not thread.is_alive(), "the server should stop once the last tab closes"


def test_refresh_reconnect_cancels_the_shutdown(session):
    grace = 1.0
    server, thread, port = _serve(
        session, exit_on_disconnect=True, disconnect_grace=grace
    )
    url = f"ws://127.0.0.1:{port}/ws"

    ws1 = ws_connect(url)  # the page loads, holding its socket
    ws1.close()  # a refresh drops it...
    ws2 = ws_connect(url)  # ...and the reload reconnects within the grace period
    time.sleep(grace * 2)  # past when the (now-cancelled) shutdown would have fired
    assert not server.should_exit, (
        "a reconnect within the grace period cancels the shutdown"
    )
    assert thread.is_alive()

    ws2.close()  # closing the reconnected tab finally stops the server
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_keep_alive_survives_a_tab_close(session):
    # exit_on_disconnect off (the --keep-alive opt-out): the server stays up.
    grace = 0.2
    server, thread, port = _serve(
        session, exit_on_disconnect=False, disconnect_grace=grace
    )
    with ws_connect(f"ws://127.0.0.1:{port}/ws"):
        pass
    time.sleep(grace * 3)
    assert not server.should_exit
    assert thread.is_alive()
    server.should_exit = True  # tidy up the daemon server
    thread.join(timeout=5)


# -- single-writer lock (multiple browsers on one session) --------------------
#
# The session is one shared EditorState, so only one connected browser -- the
# "writer" -- may edit; later browsers are read-only until they take over or the
# writer disconnects. These drive two real sockets at once (the in-process
# TestClient can't hold two live websockets), so they use the _serve helper.


def _edit_2d(view=0, point=1, x=5.0, y=6.0, frame=0):
    return {
        "type": "edit_2d",
        "view": view,
        "point": point,
        "x": x,
        "y": y,
        "frame": frame,
        "mode": "edit_2d",
    }


def test_lone_client_gets_no_role_handshake(session):
    # A single browser is the writer and is told nothing on connect, so the first
    # message it receives is still its own edit reply -- the pre-existing
    # single-client flow every in-process test relies on.
    server, thread, port = _serve(session)
    ws = ws_connect(f"ws://127.0.0.1:{port}/ws")
    try:
        ws.send(json.dumps(_edit_2d()))
        first = json.loads(ws.recv(timeout=5))
        assert first.get("type") != "role"  # no handshake ahead of the reply
        assert first["dirty"] is True
    finally:
        ws.close()
        server.should_exit = True
        thread.join(timeout=5)


def test_second_client_is_read_only(session):
    # The first browser edits; a second is read-only: its edit is refused (the
    # shared state stays clean) and it is told its role, while the writer edits fine.
    server, thread, port = _serve(session)
    url = f"ws://127.0.0.1:{port}/ws"
    writer = ws_connect(url)  # first socket -> the writer (told nothing)
    time.sleep(0.05)  # let the writer register its slot before the reader connects
    reader = ws_connect(url)  # second socket -> read-only
    try:
        role = json.loads(reader.recv(timeout=5))
        assert role["type"] == "role" and role["role"] == "reader"
        assert role["clients"] == 2

        reader.send(json.dumps(_edit_2d()))
        refused = json.loads(reader.recv(timeout=5))
        assert refused["type"] == "role" and refused["role"] == "reader"
        assert not session.state.dirty  # the read-only edit mutated nothing

        writer.send(json.dumps(_edit_2d()))
        reply = json.loads(writer.recv(timeout=5))
        assert reply.get("type") != "role"  # the writer gets the real points payload
        assert reply["dirty"] is True
    finally:
        reader.close()
        writer.close()
        server.should_exit = True
        thread.join(timeout=5)


def test_reader_can_take_over_editing(session):
    # A read-only browser claims the writer slot: it is promoted, the previous
    # writer is demoted, and the roles' edit permissions swap accordingly.
    server, thread, port = _serve(session)
    url = f"ws://127.0.0.1:{port}/ws"
    writer = ws_connect(url)
    time.sleep(0.05)
    reader = ws_connect(url)
    try:
        assert json.loads(reader.recv(timeout=5))["role"] == "reader"

        reader.send(json.dumps({"type": "claim"}))
        promoted = json.loads(reader.recv(timeout=5))
        assert promoted["type"] == "role" and promoted["role"] == "writer"
        demoted = json.loads(writer.recv(timeout=5))  # old writer pushed a demotion
        assert demoted["type"] == "role" and demoted["role"] == "reader"

        reader.send(json.dumps(_edit_2d()))  # the new writer can edit
        assert json.loads(reader.recv(timeout=5))["dirty"] is True
        writer.send(json.dumps(_edit_2d(point=2)))  # the old writer is now refused
        assert json.loads(writer.recv(timeout=5))["role"] == "reader"
    finally:
        reader.close()
        writer.close()
        server.should_exit = True
        thread.join(timeout=5)


def test_writer_disconnect_promotes_the_next_client(session):
    # When the editing browser leaves, a remaining read-only browser is promoted to
    # writer (so a surviving viewer can edit), without any action on its part.
    server, thread, port = _serve(session)
    url = f"ws://127.0.0.1:{port}/ws"
    writer = ws_connect(url)
    time.sleep(0.05)
    reader = ws_connect(url)
    try:
        assert json.loads(reader.recv(timeout=5))["role"] == "reader"
        writer.close()  # the editor closes its tab
        promoted = json.loads(reader.recv(timeout=5))
        assert promoted["type"] == "role" and promoted["role"] == "writer"
        reader.send(json.dumps(_edit_2d()))  # the promoted browser can now edit
        assert json.loads(reader.recv(timeout=5))["dirty"] is True
    finally:
        reader.close()
        server.should_exit = True
        thread.join(timeout=5)


# -- absence over the wire ----------------------------------------------------


def test_absent_rides_every_points_payload(client, session, result):
    # Absence gates whether a joint is drawn, so it must be on the LEAN mid-drag reply
    # too -- otherwise the phantom limb flashes back on for the duration of every drag.
    p = 4
    session.state.set_absent([p], True, whole_recording=True)
    for url in ("/api/points/0?mode=view", "/api/points/0?mode=view&verbose=true"):
        payload = client.get(url).json()
        absent = payload["absent"]
        assert len(absent) == result.n_views
        assert all(row[p] for row in absent)
        assert not any(row[5] for row in absent)
        # ... and the point itself is drawn nowhere
        assert all(row[p] is None for row in payload["points"])


def test_set_absent_edit_collapses_targets_to_a_point_set(session):
    from deeperfly.gui.server import _handle_edit

    s = session.state
    # A (view, point) selection spanning several views is ONE fact about the animal.
    msg = {
        "type": "set_absent",
        "targets": [[0, 4], [1, 4], [2, 4]],
        "absent": True,
        "scope": "recording",
        "frame": 0,
        "mode": "view",
    }
    payload = _handle_edit(session, msg)
    assert s.absent_mask(0)[4]
    assert "notice" in payload and "not on this animal" in payload["notice"]
    # ... and it is recording-wide, not just frame 0
    assert s.absent_mask(s.n_frames - 1)[4]
    assert payload["absent_recording"] == [4]


def test_set_absent_defaults_to_this_frame_only(session):
    from deeperfly.gui.server import _handle_edit

    s = session.state
    payload = _handle_edit(
        session,
        {
            "type": "set_absent",
            "targets": [[0, 4]],
            "absent": True,
            "frame": 1,
            "mode": "view",
        },  # no scope -> this frame
    )
    assert s.absent_mask(1)[4] and not s.absent_mask(0)[4]
    assert payload["absent_recording"] == []  # not an amputation
    assert "frame 1" in payload["notice"]


def test_edit_on_an_absent_point_is_refused_with_a_notice(session):
    from deeperfly.gui.server import _handle_edit

    s = session.state
    s.set_absent([4], True, whole_recording=True)
    payload = _handle_edit(
        session,
        {"type": "edit_2d", "view": 0, "point": 4, "x": 5.0, "y": 6.0, "mode": "view"},
    )
    assert "notice" in payload
    assert not s.labels.gt_authored[0, 0, 4]


def test_save_mirrors_absence_into_results_h5(client, session, result, tmp_path):
    # The pipeline reads `animal/` from results.h5 and never opens labels.h5, so the
    # editor's Save is what carries an authored declaration across that boundary.
    from deeperfly.results import StageStore

    results_path = tmp_path / "results.h5"
    result.save(results_path)
    session.results_path = str(results_path)
    session.state.set_absent([4], True, whole_recording=True)
    assert client.post("/api/save").status_code == 200

    absent, _ = StageStore(results_path).read_animal()
    assert absent is not None and absent[4] and not absent[5]


# -- chirality (left/right swap) ----------------------------------------------


def test_points_payload_carries_the_chirality_verdict(client):
    """The warning rides the settle/plain reply, alongside `conf` and for the same reason:
    it needs the frame's derived 3D, and a warning that flickers through a drag is worse
    than one that appears when the drag lands.
    """
    payload = client.get("/api/points/0?mode=view").json()
    v = payload["chirality"]
    assert set(v) == {"decided", "reason", "swapped", "n_pairs", "separation_frac"}
    assert isinstance(v["decided"], bool)
    # An undecided verdict must never carry candidates -- the UI would read them as findings.
    if not v["decided"]:
        assert v["swapped"] == []
        assert v["reason"]


def test_a_planted_swap_reaches_the_payload_with_names(session, client, fly):
    """End to end: swap a pair in the derived 3D and the editor is told which pair, by name.

    The names ride along with the indices because the front-end draws from the indices and
    the operator reads the names.
    """
    state = session.state
    pts3d = state.display_pts3d(0)
    assert pts3d is not None
    i, j = (int(x) for x in np.asarray(fly.symmetries)[4])
    # Write a clearly-mirrored pose so the check has a well-conditioned axis to fit, then
    # swap one pair inside it.
    half = np.linspace(-1, 1, fly.n_points // 2)
    posed = np.zeros((fly.n_points, 3))
    posed[: fly.n_points // 2] = np.stack(
        [half, np.full_like(half, 1.0), half * 0.3], 1
    )
    posed[fly.n_points // 2 :] = posed[: fly.n_points // 2] * [1, -1, 1]
    posed[[i, j]] = posed[[j, i]]
    state._pts3d_cache[0] = posed

    v = client.get("/api/points/0?mode=view").json()["chirality"]
    assert v["decided"], v["reason"]
    flagged = {tuple(s["points"]) for s in v["swapped"]}
    assert (i, j) in flagged
    names = [s["names"] for s in v["swapped"] if tuple(s["points"]) == (i, j)][0]
    assert names == [fly.point_names[i], fly.point_names[j]]


def test_the_mid_drag_reply_omits_the_verdict_rather_than_clearing_it(session):
    """`include_nmf=False` is the lean drag stream; the key must be *absent*, not null,
    so the front-end can tell "unchanged" from "now clean"."""
    from deeperfly.gui.server import _points_payload

    lean = _points_payload(session, 0, "edit_3d", include_nmf=False)
    assert "chirality" not in lean
    full = _points_payload(session, 0, "edit_3d", include_nmf=True)
    assert "chirality" in full
