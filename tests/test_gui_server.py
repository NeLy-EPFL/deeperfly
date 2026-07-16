"""Tests for the web GUI server (:mod:`deeperfly.gui.server`), driven in-process.

FastAPI's ``TestClient`` exercises the HTTP API and the edit WebSocket against a
:class:`~deeperfly.gui.session.Session` built on the synthetic 7-camera fixture
with blank frames (no footage), so there is no real server, browser, or video
decoding involved. The edit ops themselves are covered in ``test_gui.py``; these
tests check the request/response wiring on top of them.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient
from helpers import HEIGHT, WIDTH
from websockets.sync.client import connect as ws_connect

from deeperfly.gui import EditorState, FrameSource, Session
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
        corrections_path=tmp_path / "corrections.h5",
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
        corrections_path=tmp_path / "corrections.h5",
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
        corrections_path=tmp_path / "corrections.h5",
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
            corrections_path=tmp_path / "corrections.h5",
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


def test_corrected_endpoint_lists_edited_frames(client):
    """/api/corrected reports the frames carrying corrections, with point counts."""
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
    assert frames[0]["count"] == 2


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
    assert session.corrections_path.exists()
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
