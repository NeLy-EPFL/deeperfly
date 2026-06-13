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
            {"type": "toggle_invisible", "view": view, "point": point,
             "frame": 0, "mode": "edit_3d"}
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


def test_ws_reset_point_view_reverts_one_view(client):
    point = 3
    with client.websocket_connect("/ws") as ws:
        for view, xy in ((0, (12.0, 34.0)), (1, (56.0, 78.0))):
            ws.send_json(
                {"type": "edit_2d", "view": view, "point": point,
                 "x": xy[0], "y": xy[1], "frame": 0, "mode": "edit_2d"}
            )
            ws.receive_json()
        ws.send_json(
            {"type": "reset_point_view", "view": 0, "point": point, "frame": 0, "mode": "edit_2d"}
        )
        reply = ws.receive_json()
    # View 0 reverts off its edit; view 1's edit is untouched.
    assert not np.allclose(reply["points"][0][point], [12.0, 34.0])
    assert np.allclose(reply["points"][1][point], [56.0, 78.0])


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
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    holder["server"] = server
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "server did not start"
    return server, thread, port


def test_closing_the_last_tab_stops_the_server(session):
    server, thread, port = _serve(session, exit_on_disconnect=True, disconnect_grace=0.3)
    with ws_connect(f"ws://127.0.0.1:{port}/ws"):
        pass  # a browser tab opens its socket, then closes it (the tab is closed)
    thread.join(timeout=5)
    assert not thread.is_alive(), "the server should stop once the last tab closes"


def test_refresh_reconnect_cancels_the_shutdown(session):
    grace = 1.0
    server, thread, port = _serve(session, exit_on_disconnect=True, disconnect_grace=grace)
    url = f"ws://127.0.0.1:{port}/ws"

    ws1 = ws_connect(url)  # the page loads, holding its socket
    ws1.close()  # a refresh drops it...
    ws2 = ws_connect(url)  # ...and the reload reconnects within the grace period
    time.sleep(grace * 2)  # past when the (now-cancelled) shutdown would have fired
    assert not server.should_exit, "a reconnect within the grace period cancels the shutdown"
    assert thread.is_alive()

    ws2.close()  # closing the reconnected tab finally stops the server
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_keep_alive_survives_a_tab_close(session):
    # exit_on_disconnect off (the --keep-alive opt-out): the server stays up.
    grace = 0.2
    server, thread, port = _serve(session, exit_on_disconnect=False, disconnect_grace=grace)
    with ws_connect(f"ws://127.0.0.1:{port}/ws"):
        pass
    time.sleep(grace * 3)
    assert not server.should_exit
    assert thread.is_alive()
    server.should_exit = True  # tidy up the daemon server
    thread.join(timeout=5)
